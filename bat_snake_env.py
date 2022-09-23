import numpy as np
from matplotlib import pyplot as plt

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from coordinate_utils import ObjCoordinates as Obj
from coordinate_utils import BatPosition as Bat
from coordinate_utils import BatAction as Action
from coordinate_utils import BatEcho as Echo
from coordinate_utils import BatStatus as Status
from strategies import Locomotion as Loco

from render_utils import render_trajectory

from coordinate_utils import polar_bat_reference


class AvoidanceTestEnv:
    def __init__(self, preset=1, 
                 strategy_decode={'avoid':0, 'approach':1}):
        self.preset = preset
        self.obj = Obj(preset=preset)
        self.bat = Bat(preset=preset)
        self.act = Action(strategy_decode)
        self.echo= Echo()
        self.echo.call(self.obj, self.bat)
        self.status=Status()
        self.loco = Loco()
        self.visual = True

    def _reset(self):
        self.obj.reset()
        self.bat.reset()
        self.act.reset()
        self.echo.reset()
        self.status.reset()
        return None

    def _step(self, action):
        self.loco.update_strategy(action)
        self.loco.update_state(self.echo)
        self.bat.rotational_move(self.loco.turn_rate)
        self.bat.longitudinal_move(self.loco.move_rate)
        
        if self.visual:
            fig, ax1, ax2 = render_trajectory(self.obj, self.bat, self.echo, self.act, self.status, with_echo=True)
            plt.show()
            #print('bat (x,y,a)='+str(self.bat._tracker))
            #print('move='+str(self.loco.move_rate)+'  |  turn='+str(self.loco.turn_rate))
        return None

    def hit_wall_based_on_coordinates(self):
        outer_lim, inner_lim = (6.52, 4.52)
        hit = False
        x = self.bat._tracker[0,0]
        y = self.bat._tracker[0,1]

        polar = polar_bat_reference(self.bat._tracker, np.zeros((1,3))).reshape(2,)
        theta = polar[1]

        # Quadrant 1:
        if -45<=theta<45:
            if np.abs(x) > outer_lim or np.abs(x) < inner_lim:
                hit=True
        # Quadrant 2:
        if 45<=theta<135:
            if np.abs(y)>outer_lim or np.abs(y)<inner_lim:
                hit=True
        # Quadrant 3:
        if theta>=135 and theta<-135:
            if np.abs(x) > outer_lim or np.abs(x) < inner_lim:
                hit=True
        # Quadrant 4:
        if -135<=theta<-45:
            if np.abs(y)>outer_lim or np.abs(y)<inner_lim:
                hit=True

        return hit

    def _run(self):
        rewards = 0
        lives = 10
        time_limit = 1000
        strategy = 'avoid'
        success = 0 
        
        for idx in range(10):
            print('episode = '+str(idx+1)+'/1000')
            self.obj.set_new_food(idx)
            food = self.obj._coordinates[self.obj._coordinates[:,2]==1][0].reshape(1,3)
            step = 0
            for _ in range(time_limit):
                
                print('step = '+str(step)+' | rewards='+str(rewards))
                food_polar = polar_bat_reference(food, self.bat._tracker).reshape(2,)
                food_distance = food_polar[0]
                azimuth = food_polar[1] - self.bat._tracker[0,2]
                azimuth = (azimuth - 360) if azimuth>180 else azimuth + 360 if azimuth<-180 else azimuth
                in_range = food_distance < 2.5
                in_view_lim = 30 if food_distance>2.0 else -20*food_distance+70 if food_distance>1.25 else 45.0
                in_view = np.abs(azimuth) < in_view_lim

                strategy = 'approach' if in_range and in_view else 'avoid'

                # implement avoid wall!
                if self.hit_wall_based_on_coordinates():
                    strategy = 'avoid'

                if strategy=='avoid':
                    self._step(0)
                    self.act.update(self.obj, self.bat, self.echo, self.status, action_id=np.array([0]))
                if strategy=='approach':
                    self._step(1)
                    self.act.update(self.obj, self.bat, self.echo, self.status, action_id=np.array([1]))
                tt = False
                if self.status.hit == 1:
                    success += 1
                    rewards += 1
                    #self._reset()
                    break
                if self.status.hit == 2:
                    tt = True
                    lives -= 1
                    rewards -= 2
                    break
                if self.status.hit == 0:
                    rewards += 0.1*((0.8)**step)
                step +=1 
                #print('bat @ ' + str(tuple(np.round(self.bat._tracker.reshape(3,),2))) + ' | azimuth='+str(np.round(azimuth,2))+ ' | distance='+str(np.round(food_distance,2)))
                #print('strategy= '+self.loco.strategy+' | move='+str(np.round(self.loco.move_rate,2))+' | turn='+str(np.round(self.loco.turn_rate,2)))
                #print('IID='+str(np.round(self.loco.cache['iid'],2))+' | Left='+str(np.round(self.loco.cache['left'],2))+ ' | Right='+str(np.round(self.loco.cache['right'],2)) )
                #print('estimated distance = '+str(self.loco.distance2hit))
            if tt:
                if lives > 10:
                    print('Life left = '+str(lives))
                    self.bat.longitudinal_move(-0.15)
                else:
                    print('DEAD')
                    self._reset()
                    break
                    
        return success

    def check_iid(self):
        test_azi = np.linspace(-45, 45, 91)
        self.obj._coordinates = np.array([0.0, 1.0, 1], dtype=np.float64).reshape(1,3)
        self.bat._tracker = np.array([0.0, 0.0, 135], dtype=np.float64).reshape(1,3)
        iid_0 = []
        turn_0 = []
        for _ in test_azi:
            self.act.update(self.obj, self.bat, self.echo, self.status, action_id=np.array([1]))
            self.loco.update_strategy(1)
            self.loco.update_state(self.echo)
            iid_0.append(self.loco.cache['iid'])
            turn_0.append(self.loco.turn_rate)
            self.bat.rotational_move(-1.0)
        #
        self.obj._coordinates = np.array([0.0, 1.5, 1], dtype=np.float64).reshape(1,3)
        self.bat._tracker = np.array([0.0, 0.0, 135], dtype=np.float64).reshape(1,3)
        iid_1 = []
        turn_1 = []
        for _ in test_azi:
            avg_iid = 0
            avg_turn = 0
            for _ in range(20):
                self.act.update(self.obj, self.bat, self.echo, self.status, action_id=np.array([1]))
                self.loco.update_strategy(1)
                self.loco.update_state(self.echo)
                avg_iid += self.loco.cache['iid']
                avg_turn += self.loco.turn_rate
            avg_iid /= 20
            avg_turn /= 20
             
            iid_1.append(avg_iid)
            turn_1.append(avg_turn)
            self.bat.rotational_move(-1.0)
        #
        self.obj._coordinates = np.array([0.0, 2.0, 1], dtype=np.float64).reshape(1,3)
        self.bat._tracker = np.array([0.0, 0.0, 135], dtype=np.float64).reshape(1,3)
        iid_2 = []
        turn_2 = []
        for _ in test_azi:
            avg_iid = 0
            avg_turn = 0
            for _ in range(20):
                self.act.update(self.obj, self.bat, self.echo, self.status, action_id=np.array([1]))
                self.loco.update_strategy(1)
                self.loco.update_state(self.echo)
                avg_iid += self.loco.cache['iid']
                avg_turn += self.loco.turn_rate
            avg_iid /= 20
            avg_turn /= 20
             
            iid_2.append(avg_iid)
            turn_2.append(avg_turn)
            self.bat.rotational_move(-1.0)
        
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(test_azi, iid_0, label='1.0 meter distance')
        ax1.plot(test_azi, iid_1, label='1.5 meter distance')
        ax1.plot(test_azi, iid_2, label='2.0 meter distance')
        ax1.set_xlabel('azimuth')
        ax1.set_ylabel('iid')
        ax1.legend()
        ax2.plot(test_azi, turn_0, label='1.0 meter distance')
        ax2.plot(test_azi, turn_1, label='1.5 meter distance')
        ax2.plot(test_azi, turn_2, label='2.0 meter distance')
        ax2.set_xlabel('azimuth')
        ax2.set_ylabel('turn rate')
        ax2.legend()
        plt.show()
        return None


class BatSnake_base(py_environment.PyEnvironment):
    def __init__(self, preset=1, 
                 strategy_decode={'avoid':0, 'approach':1},
                 time_limit=1000, max_level=5):
        self.preset = preset
        self.obj = Obj(preset=preset)
        self.bat = Bat(preset=preset)
        self.act = Action(strategy_decode)
        self.echo= Echo()
        self.echo.call(self.obj, self.bat)
        self.status=Status()
        self.loco = Loco()

        self.visual = False

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self.act.decode)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(100,), dtype=np.float64, minimum=0, name='observation')
        self._state = self.echo._echo
        self.reward_setting = {'food': 2, 'step': [0.0, 0.8], 'collide': -2}
        self._episode_ended = False

        self.time_limit = time_limit
        self.max_level = max_level
        self.level = 0
        self.obj.set_new_food(self.level)

        self.debug = False

        self.ep_step = 0

    
    def action_spec(self):
        return self._action_spec


    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self._episode_ended = False
        self.obj.reset()
        self.bat.reset()
        self.act.reset()
        self.echo.reset()
        self.status.reset()
        self.loco.reset()
        self._state = np.copy(self.echo._echo)
        self.level = 0
        self.obj.set_new_food(self.level)

        return ts.restart(np.array([self._state], dtype=np.float64).reshape(100,))


    def _step(self, action):
        reward = 0
        if self._episode_ended:
            return self._reset()
        
        self.status.reset()

        self.loco.update_strategy(action)
        self.loco.update_state(self.echo)
        self.bat.rotational_move(self.loco.turn_rate)
        self.bat.longitudinal_move(self.loco.move_rate)

        self.act.update(self.obj, self.bat, self.echo, self.status, action_id=action)
        self._state = np.copy(self.echo._echo)
        self.status.check(self.obj, self.bat)
        
        if self.status.hit == 0:
            reward = self.reward_setting['step'][0]*(self.reward_setting['step'][1]**self.act.steps)
        if self.status.hit == 1 and np.abs(self.status.food_azimuth)<=80:
            reward = self.reward_setting['food']
            self.level += 1
            self.act.steps = 0
            self.obj.set_new_food(self.level)
        if self.status.hit == 2:
            self._episode_ended = True
            reward = self.reward_setting['collide']
        
        if (self.act.steps>self.time_limit) or (self.level > self.max_level):
            self._episode_ended = True
        
        if self.visual:

            fig, ax1, ax2 = render_trajectory(self.obj, self.bat, self.echo, self.act, self.status, with_echo=True)
            fig.savefig('ep_0_step_'+str(self.ep_step) + '.png')

        if self.debug:
            print('level = ' +str(self.level))
            print('level steps = ' + str(self.act.steps))

        self.ep_step += 1

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.float64).reshape(100,), reward=np.round(reward,4))
        else:
            return ts.transition(np.array([self._state], dtype=np.float64).reshape(100,), reward=np.round(reward,4), discount=1.0)
        

    def update_setting(self, preset=None, obj_radius=None, visual=None, time_limit=None, max_level=None, reward_setting=None):
        if preset != None:
            self.preset = preset
            self.obj.preset = preset
            self.bat.preset = preset
        if obj_radius != None:
            self.obj.radius = obj_radius
        if visual != None:
            self.visual = visual
        if time_limit != None:
            self.time_limit = time_limit
        if max_level != None:
            self.max_level = max_level
        if reward_setting != None:
            self.reward_setting = reward_setting
        return None
