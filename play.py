import os
import sys
import retro
import argparse
import numpy as np
from nn import *
from utils import *
from rominfo import *
from agent import Agent
import time

# ARGPARSER - Recebe argumentos por linha de comando #
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--show", help="mostrar state", action="store_true")
parser.add_argument("-a", "--agent", type=str, metavar='',
                    help="agente em 'agents/'", default='current')
parser.add_argument("-l", "--level", type=str, metavar='',
                    help="fase", default='YoshiIsland2')
args = parser.parse_args()

radius = 6

input_size = (2*radius + 1) * (2*radius + 1)
output_size = len(actions_list)
topology = [input_size, 64, output_size]


def play(agent, printstate, level):
    """ O agente joga a fase. A simulação é interrompida quando o agente
    vence o level, morre, ou fica parado sem ter progresso por muito tempo.
    Após o término da simulação, o score do agente é calculado e ele
    é retornado.
    """

    env = retro.make(game='SuperMarioWorld-Snes', state=level, players=1)
    env.mode = 'normal'

    points = 0

    try:
        t = time.time()
        env.reset()

        while True:
            ram = getRam(env)
            state, x, y = getState(ram, radius)

            if printstate:
                printState(state, radius)

            agent.setPos(x, y)
            
            food = [state]
            act_idx = np.argmax(agent.brain.predict(food))
            action = actions_list[act_idx]

            # Agente bate em bloco de mensagem
            if ram[0x1426] != 0:
                performAction(1, env)
                action = 0

            is_dead = agent.is_stoped() or ram[0x0DDA] == 0xff
            # Agente morreu sem chegar ao fim da fase
            if is_dead and ram[0x1493] == 0x00:
                break

            reward, done, info, new_t = performActionWithDelay(action, env, t)

            points += reward

            env.render()
            
            t = new_t

    finally:
        env.render(close=True)
        env.close()

    agent.points = points
    agent.setScore()
    return agent


def load_player(play_file):
    """ Carrega o agente salvo em 'play_file', ou cria um novo
    agente caso esse não exista
    """
    if os.path.isfile(play_file):
        return Agent.load(play_file)
    else:
        print(f"There is no player stored in '{play_file}'.")
        return Agent(topology)


def main():

    printstate = args.show
    level = args.level
    play_file = "agents/"+args.agent+".pkl"
    agent = load_player(play_file)
    agent.reset()

    play(agent, printstate, level)
    print("Agent Report")
    print("Fitness: {:.3f} | Points: {:6,.0f} | Distance: {:4,.0f}".format(
        agent.fitness, agent.points, agent.max_x))
    print("---------------------------------------------------")

def maybeDelay(t):
    new_t = time.time()
    dt = t - new_t
    if  dt > 0:
        time.sleep(dt)
    return t + period

def performActionWithDelay(a, env, t):
    reward = 0
    bin_a = dec2bin(a)
    if a in long_actions:
        for it in range(8):
            ob, rew, done, info = env.step(bin_a)
            t = maybeDelay(t)
            reward += rew
           
    elif a in short_actions:
        for it in range(4):
            ob, rew, done, info = env.step(bin_a)
            t = maybeDelay(t)
            reward += rew
    else:
        ob, rew, done, info = env.step(bin_a)
        t = maybeDelay(t)
        reward += rew
    return reward, done, info, t

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Exit")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
