import os

import tensorflow as tf
tf.autograph.set_verbosity(0)
from tensorflow import saved_model
from bat_snake_env import BatSnake_base
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from render_utils import render_trajectory
from PIL import Image

from tf_agents.environments import tf_py_environment



NUMBER_OF_EPISODES = 2000
AGENT_ID  = 'hunt_10.29.21'
TIME_LIMIT = 600
RENDER_GIF = True
PRESET = 100
PICKLE_PATH = 'agent_checkpoints/' + AGENT_ID
DATE = '09.16.2022'
ISOLATED_TEST = True
MAZE = 'box'
MAX_LEVEL = 0

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

"""
WORK ON THE RENDERING !! TEST! THEN GOOD TO GO
"""

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
                   duration=5, loop=0)

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
        if x>4 and y>4: out = 1
        elif x<-4 and y<-4: out=2
        else: out = 0
    return out


def run_an_episode(py_e, tf_e, policy, episode=0):
    # List of things to keep track
    score = 0
    returns = 0
    bats = np.array([]).reshape(0,3)
    foods = np.array([]).reshape(0,2)
    echoes = np.array([]).reshape(0,100)
    strategies = np.array([]).reshape(0,1)
    IIDs = np.array([]).reshape(0,1)
    moves = np.array([]).reshape(0,1)
    turns = np.array([]).reshape(0,1)
    value_layers = np.array([]).reshape(0,2)
    hidden0 = np.array([]).reshape(0,128)
    hidden1 = np.array([]).reshape(0,128)
    hidden2 = np.array([]).reshape(0,128)
    hidden3 = np.array([]).reshape(0,64)
    time_step = tf_e._reset()
    i = 0
    while not time_step.is_last():
        # track prior to action:
        bats = np.vstack((bats, py_e.bat._tracker))
        if np.sum(py_e.obj._coordinates[:,2]==1) > 0:
            temp_food = py_e.obj._coordinates[py_e.obj._coordinates[:,2]==1][:,:2]
        else:
            temp_food = py_e.obj._coordinates[0,:2].reshape(1,2)
        foods = np.vstack((foods, temp_food))
        echoes = np.vstack((echoes, py_e.echo._echo))
        value_est, netcache = get_value_layer(py_e.echo._echo, policy)
        value_layers = np.vstack((value_layers, value_est))
        hidden0 = np.vstack((hidden0, netcache['0'].numpy().reshape(1,128)))
        hidden1 = np.vstack((hidden1, netcache['1'].numpy().reshape(1,128)))
        hidden2 = np.vstack((hidden2, netcache['2'].numpy().reshape(1,128)))
        hidden3 = np.vstack((hidden3, netcache['3'].numpy().reshape(1, 64)))
        # take the action
        action_step = policy.action(time_step)
        time_step = tf_e._step(action_step.action)
        # track after action
        if py_e.status.hit == 1 and np.abs(py_e.status.food_azimuth)<=45: # hit a food:
            score += 1
        elif py_e.status.hit == 1: score += 0.1
        elif py_e.status.hit == 2:
            score -=1
        returns += time_step.reward.numpy()[0] # add the returns of current steps
        strategies = np.vstack((strategies, action_step.action.numpy()))
        IIDs = np.vstack((IIDs, py_e.loco.cache['iid']))
        moves = np.vstack((moves, py_e.loco.move_rate))
        turns = np.vstack((turns, py_e.loco.turn_rate))
        if RENDER_GIF:
            render_step_to_png(i, py_e.obj, py_e.bat, py_e.echo, 
                               py_e.act, py_e.status, with_echo=True)
        i += 1
        if score==0.1: break 
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
                #score = 0
                #returns = 0
                #bats = np.array([]).reshape(0,3)
                #foods = np.array([]).reshape(0,2)
                #echoes = np.array([]).reshape(0,100)
                #strategies = np.array([]).reshape(0,1)
                #IIDs = np.array([]).reshape(0,1)
                #moves = np.array([]).reshape(0,1)
                #turns = np.array([]).reshape(0,1)
                #value_layers = np.array([]).reshape(0,2)
                #time_step = tf_e._reset()
                #i=0
                #continue
        
                
    records = {'score': score, 'returns': returns, 'bats': bats, 'foods': foods,
               'echoes': echoes, 'strategies': strategies, 'IIDs': IIDs,
               'moves': moves, 'turns': turns, 'value_layers': value_layers,
               'hidden0': hidden0, 'hidden1': hidden1, 'hidden2': hidden2, 'hidden3': hidden3 }

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
    iids_ls, moves_ls, turns_ls, value_layers_ls = ([], [], [], [])
    hidden0_ls, hidden1_ls, hidden2_ls, hidden3_ls= ([],[],[],[])
    miss_ls, wrongway_ls = ([],[])

    success_score = 0
    hit_score = 0
    miss_score = 0
    wrongway_score = 0
    hitfood_score = 0

    for episode in range(NUMBER_OF_EPISODES):
        rec = run_an_episode(py_env, tf_env, policy, episode=episode)
        episodes_ls.append(episode + 1)
        obstacles_ls.append(py_env.obj._coordinates[py_env.obj._coordinates[:,2]==2])
        scores_ls.append(rec['score'])
        returns_ls.append(rec['returns'])
        bats_ls.append(rec['bats'])
        foods_ls.append(rec['foods'])
        echoes_ls.append(rec['echoes'])
        strategies_ls.append(rec['strategies'])
        iids_ls.append(rec['IIDs'])
        moves_ls.append(rec['moves'])
        turns_ls.append(rec['turns'])
        value_layers_ls.append(rec['value_layers'])
        hidden0_ls.append(rec['hidden0'])
        hidden1_ls.append(rec['hidden1'])
        hidden2_ls.append(rec['hidden2'])
        hidden3_ls.append(rec['hidden3'])

        if ISOLATED_TEST:
            miss_ls.append(rec['miss'])
            wrongway_ls.append(rec['wrongway'])

        print('progress >> \t' +str(episode+1) +'/'+str(NUMBER_OF_EPISODES), end='\t')
        # report ending:
        if rec['score'] > 0.5:
            print('SUCCESS')
            success_score += 1
        if rec['score'] ==0.1:
            print('HIT FOOD')
            hitfood_score+=1
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

    print('DATA TABULATION COMPLETED!')
    print('SUCCESS = '+ str(success_score))
    print('HIT FOOD = '+str(hitfood_score))
    print('HIT     = ' + str(hit_score))
    print('MISS    = ' + str(miss_score))
    print('WRONGWAY= ' + str(wrongway_score))

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
        'moves': moves_ls,
        'turns': turns_ls,
        'value_layers': value_layers_ls,
        'hidden0': hidden0_ls,
        'hidden1': hidden1_ls,
        'hidden2': hidden2_ls,
        'hidden3': hidden3_ls
        })

    df.to_pickle(PICKLE_PATH +'/run_'+ DATE +'.pkl')
