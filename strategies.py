from operator import le
import numpy as np
from numpy.core.function_base import logspace

from echo_lib.echo_transform_utils import get_background, echo2envelope_z, quiet_envelope

"""
This library formulate different strategy of the bat used for locomotion.

A locomotion class with these variables:
- strategy
- echo L
- echo R
- move_rate --> distance to move forward each step
- turn rate --> angle of turning each step.
- distance to hit --> estimated distance to the closest object

bat's default velocity is 2m/s and 341d/s. default call rate is 50Hz.
-> move 0.04 meter per step.
-> turn 6.8 degrees per step.
---> round up the movement to 0.05 meter, and 5 degrees, per step.
---> That's equal to 2.5 m/s forward velocity, and 250 angular velocity.

Let's set the default threshold at 1 meter. 
When distance to hit is less than 1 meter:
- forward velocity scale down. [linearly]. 0.02 meter per step --> 1m/s
- angular velocity scale up (for avoidance strategy) max of 20 degrees per step. 
               and scale down (for avoidance strategy) min of 2 degrees per step.
               may be I can scale this with exponential decay.

"""


def call_strategy(strategy, distance, l_window, r_window, max_turn, min_move):
    left_sum = np.sum(l_window)
    right_sum = np.sum(r_window)
    if left_sum!=0 and right_sum !=0:
        iid = 10*np.log10(left_sum/right_sum)
    elif left_sum==0 and right_sum!=0:
        iid = -10
    elif right_sum==0 and left_sum!=0:
        iid = 10
    elif right_sum==0 and left_sum==0:
        iid = 0.1*np.random.rand()-0.05
    else:
        raise ValueError('iid is out of expected range!')

    if strategy == 'avoid':
        if distance >1:
            move_rate = 0.05
        elif distance >0.5:
            move_rate = (0.05 - min_move)*distance + min_move
        else:
            move_rate = 0.001
        
        turn_rate = max_turn - 500*(move_rate - min_move)
        turn_rate = turn_rate*((-1)*np.sign(iid))
        if right_sum==0 and left_sum==0:
            turn_rate = 10.0*np.random.rand() - 5.0
    elif strategy == 'approach':
        move_rate = (0.05 - min_move)*distance + min_move if distance <1 else 0.05
        turn_rate = 2
        turn_rate = turn_rate * iid
        turn_rate = np.sign(turn_rate) * max_turn if np.abs(turn_rate)>max_turn else turn_rate

    else:
        raise ValueError('only accept `avoid` and `approach` as strategies for now')

    cache = {'iid': iid, 'left': left_sum, 'right': right_sum}

    return move_rate, turn_rate, cache


class Locomotion:
    def __init__(self, strategies=None, glomax = 5.73, window_size = 10):
        self.strategies_ls = ['avoid', 'approach'] if strategies==None else strategies
        self.strategy = self.strategies_ls[0]
        self.move_rate = 0.05 # meter per step
        self.turn_rate = 5 # degrees per step
        self.l_echo = np.zeros(50)
        self.r_echo = np.zeros(50)
        self.min_move_rate = 0.02 # meter per step
        self.max_turn_rate = 20 # degrees per step
        self.distance_reference = np.floor(100*np.linspace(0.04,3.51,50))/100
        bg = quiet_envelope(echo2envelope_z(get_background('x')))
        self.l_echo_bg = bg['left'][:50]
        self.r_echo_bg = bg['right'][:50]
        self.onset_threshold = 1/glomax
        self.onset_idx = len(self.l_echo) - 1
        self.distance2hit = self.distance_reference[self.onset_idx]
        self.l_window = np.zeros(window_size)
        self.r_window = np.zeros(window_size)

        self.cache = None


    def reset(self):
        self.move_rate = 0.05 # meter per step
        self.turn_rate = 5 # degrees per step
        self.l_echo = np.zeros(50)
        self.r_echo = np.zeros(50)
        bg = quiet_envelope(echo2envelope_z(get_background('x')))
        self.l_echo_bg = bg['left'][:50]
        self.r_echo_bg = bg['right'][:50]
        self.onset_idx = len(self.l_echo) - 1
        self.distance2hit = self.distance_reference[self.onset_idx]
        window_size = len(self.l_window)
        self.l_window = np.zeros(window_size)
        self.r_window = np.zeros(window_size)
        return None


    def update_strategy(self, strategy_id):
        self.strategy = self.strategies_ls[strategy_id]
        return self.strategy


    def update_state(self, echo): # echo, distance2hit, move_rate, turn_rate
        self.l_echo = echo._echo[:50]
        self.r_echo = echo._echo[50:]
        
        ll = self.l_echo - self.l_echo_bg
        rr = self.r_echo - self.r_echo_bg
        l_onset = np.argmax(ll> self.onset_threshold)
        r_onset = np.argmax(rr> self.onset_threshold)
        l_onset = 49 if l_onset==0 else l_onset
        r_onset = 49 if r_onset==0 else r_onset

        self.onset_idx = np.min([ l_onset, r_onset ])
        self.distance2hit = self.distance_reference[self.onset_idx]
        if self.onset_idx >= 50-len(self.l_window):
            self.l_window = ll[(50-len(self.l_window)):50]
            self.r_window = rr[(50-len(self.r_window)):50]
        else:
            self.l_window = ll[self.onset_idx:(self.onset_idx + len(self.l_window))]
            self.r_window = rr[self.onset_idx:(self.onset_idx + len(self.r_window))]
        self.move_rate, self.turn_rate, self.cache = call_strategy(self.strategy, 
                                                       self.distance2hit, 
                                                       self.l_window, 
                                                       self.r_window, 
                                                       self.max_turn_rate, 
                                                       self.min_move_rate)
        return None