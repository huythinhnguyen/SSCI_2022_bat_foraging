import os

import tensorflow as tf
from tensorflow import saved_model
from bat_snake_env_MonteCarlo import BatSnake_base
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from render_utils import render_trajectory
from PIL import Image

from tf_agents.environments import tf_py_environment
NUMBER_OF_EPISODES = 10_000
AGENT_ID  = 'hunt_10.29.21'
TIME_LIMIT = 500
RENDER_GIF = True
PRESET = 1
PICKLE_PATH = 'agent_checkpoints/' + AGENT_ID
DATE = '03.03.2022'
ISOLATED_TEST = True
MAZE = 'box'
MAX_LEVEL = 0

# NORMAL DISTRIBUTION OF MONTE CARLO PARAMETER:
GAMMA_MEAN = 1
GAMMA_STD = 0.1
DELTA_MEAN = 1
DELTA_STD = 0.05

WHEELBASE = 0.25

DEBUG = False

TEMP_PNG_PATH = os.path.join(os.getcwd(),'agent_checkpoints/' + AGENT_ID + '/temp_pics')

def load_policy(agent_id, agent_dir ='agent_checkpoints/'):
    policy_dir = os.path.join(os.getcwd(), agent_dir + agent_id)
    print(policy_dir)
    policy = saved_model.load(policy_dir)
    return policy


def get_value_layer(inp, policy):
    inp = tf.convert_to_tensor(inp.astype(np.float32).reshape(1, 100))
    cache = {'0': None,
             '1': None,
             '2': None,
             '3': None,
             '4': None}
    out = tf.identity(inp)
    for i in range(5):
        m = i*2
        n = m+1
        with tf.device('/gpu:0'):
            out = tf.matmul(out, policy.model_variables[m])
            out = tf.add(out, policy.model_variables[n])
            out = tf.keras.activations.relu(out)
            cache[str(i)]= out
    value_layer = cache['4'].numpy().reshape(1,2)
    return value_layer, cache


def render_step_to_png(step, obj, bat, echo, act, status, with_echo=True):
    path = 'agent_checkpoints/' + AGENT_ID + '/temp_pics'
    if not os.path.isdir(path):
        os.mkdir(path)
    
    fig, _, _ = render_trajectory(obj, bat, echo, act, status, with_echo=with_echo)
    filename = path + '/step_' + str(step) + '.png'
    fig.savefig(filename)
    plt.close()
    return None


def render_episode_to_gif(episode):
    path = os.path.join(os.getcwd(),'agent_checkpoints/' + AGENT_ID + '/gif')
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, DATE)
    if not os.path.isdir(path):
        os.mkdir(path)

    frames = []
    imgs = os.listdir(TEMP_PNG_PATH)
    for i in range(len(imgs)):
        png_file = TEMP_PNG_PATH + '/step_' + str(i) + '.png'
        new_frame = Image.open(png_file)
        frames.append(new_frame)
    gif_file  = path + '/ep_' + str(episode) + '.gif'
    frames[0].save(gif_file, format='GIF',
                   append_images=frames[1:], save_all=True, 
                   duration=2, loop=0)
    return None


def delete_temp_png():
    import shutil
    if os.path.isdir(TEMP_PNG_PATH):
        shutil.rmtree(TEMP_PNG_PATH)
    return None


def out_of_bound(environment):
    bat = environment.bat._tracker[0,:2].reshape(2,)
    if MAZE == 'donut':
        theta  = np.round( np.degrees( np.arctan2(bat[1], bat[0]) ) , 2)
        out = 1 if theta > 45 else 2 if theta <-135 else 0
    if MAZE == 'box':
        x, y = (bat[0], bat[1])
        out = 1 if (x>0 and y>4) else 2 if (x<-4 and y<0) else 0
    return out


def run_an_episode(py_e, tf_e, policy, episode=0):
    # List of things to keep track
    score = 0
    returns = 0
    bats = np.array([]).reshape(0,3)
    foods = np.array([]).reshape(0,2)
    echoes = np.array([]).reshape(0,100)
    strategies = np.array([]).reshape(0,1)
    onset_distances = np.array([]).reshape(0,1)
    IIDs = np.array([]).reshape(0,1)
    moves = np.array([]).reshape(0,1)
    turns = np.array([]).reshape(0,1)
    value_layers = np.array([]).reshape(0,2)
    # Monte Carlo para:
    gamma = np.random.normal(GAMMA_MEAN, GAMMA_STD, (2,))
    delta = np.random.normal(DELTA_MEAN, DELTA_STD, (2,))

    # track of the estimates:
    est_moves = np.array([]).reshape(0,1)
    est_turns = np.array([]).reshape(0,1)
    
    time_step = tf_e._reset()

    # Update gamma, delta:
    py_e.loco.update_MC_para(gamma, delta)
    # track wheelbase
    wheelbase = py_e.loco.wheelbase
    i = 0

    knock_food = False
    
    if DEBUG:
        print('gamma_l='+ str(py_e.loco.gamma_l) +', gamma_r='+str(py_e.loco.gamma_r))
        print('delta_l='+ str(py_e.loco.delta_l) + ', delta_r='+str(py_e.loco.delta_r))
        
    while not time_step.is_last():
        # track prior to action:
        bats = np.vstack((bats, py_e.bat._tracker))
        if np.sum(py_e.obj._coordinates[:,2]==1) > 0:
            temp_food = py_e.obj._coordinates[py_e.obj._coordinates[:,2]==1][:,:2]
        else:
            temp_food = py_e.obj._coordinates[0,:2].reshape(1,2)
        foods = np.vstack((foods, temp_food))
        echoes = np.vstack((echoes, py_e.echo._echo))
        value_est, _ = get_value_layer(py_e.echo._echo, policy)
        value_layers = np.vstack((value_layers, value_est))
        # take the action
        action_step = policy.action(time_step)
        time_step = tf_e._step(action_step.action)
        # track after action
        if py_e.status.hit == 1 and np.abs(py_e.status.food_azimuth) <= 45: # hit a food:
            score += 1
        if py_e.status.hit == 1 and np.abs(py_e.status.food_azimuth) > 45:
            knock_food = True
        if py_e.status.hit == 2:
            score -=1
        returns += time_step.reward.numpy()[0] # add the returns of current steps
        strategies = np.vstack((strategies, action_step.action.numpy()))
        IIDs = np.vstack((IIDs, py_e.loco.cache['iid']))
        onset_distances = np.vstack((onset_distances, py_e.loco.distance2hit))
        moves = np.vstack((moves, py_e.loco.move_rate))
        turns = np.vstack((turns, py_e.loco.turn_rate))

        est_moves = np.vstack((est_moves, py_e.loco.cache['move']))
        est_turns = np.vstack((est_turns, py_e.loco.cache['turn']))

        if DEBUG:
            print(str(py_e.loco.strategy)+', move='+str(np.round(py_e.loco.move_rate,2))+', turn='+str(np.round(py_e.loco.turn_rate,2))+
                  '[move_est, turn est]='+str(np.round(np.array([py_e.loco.cache['move'], py_e.loco.cache['turn']]),2)))
        
        if ISOLATED_TEST:
            miss = False
            wrongway = False
            wormhole = out_of_bound(py_e)
            if wormhole==1:
                miss = True
                break
            if wormhole == 2:
                wrongway = True
                break

        if RENDER_GIF:
            render_step_to_png(i, py_e.obj, py_e.bat, py_e.echo, 
                               py_e.act, py_e.status, with_echo=True)
        i += 1

    time_out = False if i < TIME_LIMIT else True
        
    records = {'score': score, 'returns': returns, 'bats': bats, 'foods': foods,
               'echoes': echoes, 'strategies': strategies, 'IIDs': IIDs, 'onset_distances': onset_distances,
               'moves': moves, 'turns': turns, 'value_layers': value_layers, 
               'gamma': gamma, 'delta': delta, 'wheelbase': wheelbase,
               'est_moves': est_moves, 'est_turns': est_turns, 'knock_food': knock_food, 'time_out': time_out}

    if ISOLATED_TEST:
        records['miss'] = miss
        records['wrongway'] = wrongway

    if RENDER_GIF:
        render_episode_to_gif(episode)
        delete_temp_png()

    return records


if __name__ == '__main__':
    policy = load_policy(AGENT_ID)
    py_env = BatSnake_base(preset=PRESET, time_limit=TIME_LIMIT, max_level=MAX_LEVEL)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    
    episodes_ls, obstacles_ls, scores_ls, returns_ls = ([],[],[],[])
    bats_ls, foods_ls, echoes_ls, strategies_ls = ([], [], [], [])
    iids_ls, moves_ls, turns_ls, value_layers_ls, onset_distances_ls = ([], [], [], [], [])
    miss_ls, wrongway_ls = ([],[])
    gamma_ls, delta_ls, wheelbase_ls = ([], [], [])
    moves_est_ls, turns_est_ls = ([], [])
    knock_food_ls = []
    
    success_score = 0
    hit_score = 0
    miss_score = 0
    wrongway_score = 0

    episode = 0

    while episode < NUMBER_OF_EPISODES:
        rec = run_an_episode(py_env, tf_env, policy, episode=episode)
        if rec['time_out']:
            continue
        episodes_ls.append(episode + 1)
        obstacles_ls.append(py_env.obj._coordinates[py_env.obj._coordinates[:,2]==2])
        scores_ls.append(rec['score'])
        returns_ls.append(rec['returns'])
        bats_ls.append(rec['bats'])
        foods_ls.append(rec['foods'])
        echoes_ls.append(rec['echoes'])
        strategies_ls.append(rec['strategies'])
        iids_ls.append(rec['IIDs'])
        onset_distances_ls.append(rec['onset_distances'])
        moves_ls.append(rec['moves'])
        turns_ls.append(rec['turns'])

        moves_est_ls.append(rec['est_moves'])
        turns_est_ls.append(rec['est_turns'])
        
        value_layers_ls.append(rec['value_layers'])

        if ISOLATED_TEST:
            miss_ls.append(rec['miss'])
            wrongway_ls.append(rec['wrongway'])

        gamma_ls.append(rec['gamma'])
        delta_ls.append(rec['delta'])
        wheelbase_ls.append(rec['wheelbase'])

        knock_food_ls.append(rec['knock_food'])
        
        print('progress >> \t' +str(episode+1) +'/'+str(NUMBER_OF_EPISODES), end='\t')

        # report ending:
        if rec['score'] > 0:
            print('SUCCESS')
            success_score += 1
        if rec['score'] < 0:
            print('HIT WALL')
            hit_score += 1
        if rec['score'] == 0:
            if rec['miss']:
                print('MISS')
                miss_score +=1
            if rec['wrongway']:
                print('WRONG-WAY')
                wrongway_score +=1

        if episode%500==0:
            df = pd.DataFrame({
                'episodes': episodes_ls,
                'obstacles': obstacles_ls,
                'scores': scores_ls,
                'returns': returns_ls,
                'bats': bats_ls,
                'foods': foods_ls,
                'echoes': echoes_ls,
                'strategies': strategies_ls,
                'iids': iids_ls,
                'onset_distances': onset_distances_ls,
                'moves': moves_ls,
                'turns': turns_ls,
                'moves_est': moves_est_ls,
                'turns_est': turns_est_ls,
                'value_layers': value_layers_ls,
                'miss': miss_ls,
                'wrongway': wrongway_ls,
                'gamma': gamma_ls,
                'delta': delta_ls,
                'wheelbase': wheelbase_ls,
                'knock_food': knock_food_ls })
            df.to_pickle(PICKLE_PATH + '/run_'+DATE+'e'+str(episode)+'checkpoint.pkl')
            print('checkpoint at episode = '+str(episode)+ '  SUCCESS='+str(success_score))
            del df

        episode += 1

    df = pd.DataFrame({
        'episodes': episodes_ls,
        'obstacles': obstacles_ls,
        'scores': scores_ls,
        'returns': returns_ls,
        'bats': bats_ls,
        'foods': foods_ls,
        'echoes': echoes_ls,
        'strategies': strategies_ls,
        'iids': iids_ls,
        'onset_distances': onset_distances_ls,
        'moves': moves,
        'turns': turns,
        'moves_est': moves_est,
        'turns_est': turns_est,
        'value_layers': value_layers_ls,
        'miss': miss_ls,
        'wrongway': wrongway_ls,
        'gamma': gamma_ls,
        'delta': delta_ls,
        'wheelbase': wheelbase_ls,
        'knock_food': knock_food_ls })

    df.to_pickle(PICKLE_PATH + '/run_' + DATE + '.pkl')
    
    print('DATA TABULATION COMPLETED!')
    print('SUCCESS = '+ str(success_score))
    print('HIT     = ' + str(hit_score))
    print('MISS    = ' + str(miss_score))
    print('WRONGWAY= ' + str(wrongway_score))
