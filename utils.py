# Based on utils.py
# utility functions for the Q-Learning agent
# Author: Fabrício Olivetti de França

# Changed by: Allef Kristian Tavares Rocha

import os
import numpy as np
from collections import defaultdict

INF = float('inf')

fps = 10
period = 1/fps

# Todas as possíveis ações
actions_map = {'noop':0, 'down':32, 'up':16, 'jump':1, 'hold':2, 
               'left':64, 'jumpleft':65, 'runleft':66, 'runjumpleft':67, 
               'right':128, 'jumpright':129, 'runright':130, 'runjumpright':131, 
               'spin':256, 'spinright':384, 'runspinright':386, 'spinleft':320, 'spinrunleft':322
               }

# Vamos usar apenas um subconjunto
actions_list  = [66,128,130,131]
long_actions  = [131]
short_actions = [act for act in actions_list if act not in long_actions]


mm = defaultdict(lambda: "??")
mm.update({0: '  ', 1: '$$', -1: '@@'})
convert = np.vectorize(lambda x: mm[x])

def dec2bin(dec):
    return list("{0:b}".format(dec))[::-1]
  
def printState(state, radius):
    rstate = np.reshape(state, (2*radius + 1, 2*radius + 1))
    _ = os.system("clear")
    convState = convert(rstate)
    convState[radius+1][radius] = "XX"
    out = ''
    for i, line in enumerate(convState):
        out += ''.join(line)
        out += '|\n'
    out += '_'*(4*radius+2) + '|\n'
    print(out)


def performAction(a, env):
    reward = 0
    bin_a = dec2bin(a)
    if a in long_actions:
        for it in range(8):
            ob, rew, done, info = env.step(bin_a)
            reward += rew
    elif a in short_actions:
        for it in range(4):
            ob, rew, done, info = env.step(bin_a)
            reward += rew
    else:
        ob, rew, done, info = env.step(bin_a)
        reward += rew
    return reward, done, info


def scaling(value, old_min, old_max, new_min, new_max):
  return ((value - old_min)/(old_max - old_min))*(new_max-new_min) + new_min