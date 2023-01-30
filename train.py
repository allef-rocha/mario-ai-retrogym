import os
import sys
import retro
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from nn import *
from utils import *
from rominfo import *
from agent import Agent

# argparser - Recebe argumentos por linha de comando
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-H", "--hide"        , help="esconder tela do jogo",   action="store_true")
# parser.add_argument("-m", "--multiprocess", help="usar multiprocessamento", action="store_true")
parser.add_argument("-s", "--startover"   , help="recomeçar treinamento",   action="store_true")
parser.add_argument("-a", "--agent"       , type=str , metavar='', help="agente em 'agents/'"  , default='current')
parser.add_argument("-n", "--numproc"     , type=int, metavar='',  help="numero de processos"  , default=1)
parser.add_argument("-g", "--generations" , type=int , metavar='', help="numero de gerações"   , default=0)
parser.add_argument("-l", "--level"       , type=str , metavar='', help="fase"                 , default='YoshiIsland2')
parser.add_argument("-p", "--popsize"     , type=int , metavar='', help="tamanho da população" , default=50)

args = parser.parse_args()

radius      = 6

input_size  = (2*radius + 1) * (2*radius + 1) # mesmo tamanho do state (rominfo)
output_size = len(actions_list)
topology    = [input_size, 64, output_size]

best_ever  = None
max_score =  6000 # não deve passar disso

ALPHA = 0.2 # Constante que influencia desvio padrão da mutação



def populate(training_file, topology, popsize):
    """ Cria uma população a partir de um agente salvo, caso esse exista,
    ou cria uma população do zero.
    """
    if not args.startover and os.path.isfile(training_file):
        global best_ever
        loaded = Agent.load(training_file)
        best_ever = loaded

        agents = [loaded.copy() for i in range(popsize)]
        
        temperature = (1 - loaded.fitness)
        stdDev = max(temperature * ALPHA, 0.01)

        for agent in agents:
            agent.mutate(prob=MUT_RATE, stdDev=stdDev)
    else:
        agents = [Agent(topology=topology) for i in range(popsize)]

    return agents



def train(agents, generation, processes, render, level):
    """ Cria um processo para cada agente. O numero de processos simultâneos
    é limitado pela entrada 'processes'
    """
    popsize = len(agents)
    completed = []
    with ProcessPoolExecutor(max_workers=processes) as executor:
            try:
                futures = [executor.submit(train_level, agent=agent, render=render, level=level)
                        for agent in agents]
                
                print("Generation: {0:}".format(generation))
                pbar = tqdm(total=popsize, colour="green") # barra de progresso
                for future in as_completed(futures):
                    completed.append(future.result())
                    pbar.update(1)
                pbar.close()

            finally:
                for process in executor._processes.values():
                    process.kill()
    return completed



def train_level(agent, render, level):
    """ Treina o agente. O treinamento é interrompido quando o agente
    vence o level, morre, ou fica parado sem ter progresso por muito tempo.
    Após o término do treinamento, o score do agente é calculado e ele
    é retornado.
    """

    env = retro.make(game='SuperMarioWorld-Snes',
                     state=level, players=1)
    env.mode = 'fast'

    # Se a posição do agente seja ultrapassada pela dead line, ele morre
    dead_line = -300
    dead_line_speed = 5

    points = 0
    
    try:
        env.reset()

        while True:
            ram = getRam(env)
            state, x, y = getState(ram, radius)
            # printState(state, radius)
            agent.setPos(x, y)
            
            nn_input = [state]
            prediction = agent.brain.predict(nn_input)
            act_idx = np.argmax(prediction)
            action = actions_list[act_idx]

            # Agente bate em bloco de mensagem
            if ram[0x1426] != 0:
                performAction(1, env)
                action = 0

            # ram[0x0DDA] = 0xff indica que o agente morreu ou concluiu a fase
            is_done = agent.is_stoped() or x < dead_line or ram[0x0DDA] == 0xff

            # ram[0x1493] = 0x00 indica que a animação de término da fase já acabou
            if is_done and ram[0x1493] == 0x00:
                break
                
            dead_line += dead_line_speed

            reward, done, info = performAction(action, env)

            points += reward

            if(render):
                env.render()

    finally:
        env.render(close=True)
        env.close()

    agent.points = points
    agent.setScore()
    return agent



def repopulate(agents, training_file):
    """ Refaz o população baseada no desempenho da população anterior.
    Aqui o fitness de cada agente é calculado.
    Quanto maior o fitness do agente, maior a probabilidade de ele gerar um filho.
    Quanto maior o fitness do agente, menor é a probabilidade de um link de sua rede neural
    sofrer mutação. 
    """
    popsize = len(agents)
    agents.sort(reverse=True)

    # Fitness baseado na pontuação
    fitnesses = list([a.score/max_score for a in agents])

    for i,a in enumerate(agents):
        a.fitness = fitnesses[i]

    # Probabilidade normalizada, baseada no fitness
    fitSum = sum(fitnesses)

    # Probabilidade de reprodução (soma das probabilidades = 1)
    prob = [c/fitSum for c in fitnesses]

    # Atualiza o best_ever, caso necessário
    curr_best = agents[0]
    global best_ever
    if best_ever is None or curr_best > best_ever:
        best_ever = curr_best
        curr_best.save(training_file)
        print("BEST ONE REPLACED!")
    
    children = []

    for i in range(popsize):
        p = np.random.choice(agents, p=prob).copy()
        children.append(p)

    for child in children:

        temperature = 1 - child.fitness
        stdDev = max(temperature * ALPHA, 0.01)

        child.mutate(prob=MUT_RATE, stdDev=stdDev)

    # Imprime relatório do melhor agenta da geração
    print("Fitness: {:.3f} | Points: {:4,.0f} | Distance: {:4,.0f}".format(
        curr_best.fitness, curr_best.points, curr_best.max_x))
    print("------------------------------------------------")

    return children



def main():
    # Extração dos dados obtidos através das flags (argparse)
    generations = args.generations if args.generations != 0 else float('inf')
    popsize     = args.popsize
    processes   = min(os.cpu_count() - 2, args.numproc)
    render      = not args.hide
    level       = args.level
    training_file = "agents/"+args.agent+".pkl"
    
    agents = populate(training_file, topology, popsize)

    generation = agents[0].generation
    generations += generation

    # loop principal. Para interromper: CTRL + C
    while generation < generations:
        eval = train(agents, generation, processes, render, level)
        agents = repopulate(eval, training_file)
        generation += 1




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
