from nn import MUT_AMMOUNT, MUT_RATE, NeuralNetwork
from functools import total_ordering
import pickle

STOP_TIME = 60

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def set(self, x, y):
        self.x = x
        self.y = y
    
@total_ordering
class Agent:
    def __init__(self, topology:list):
        self.brain = NeuralNetwork(topology)

        self.generation   = 1
        
        self.fitness      = 0 # [0,1]
        self.points       = 0 # pontos coletados na fase
        self.score        = 0 # pontuação final: depende também da distância

        self.curr_pos     = Vec2(0,0)
        self.prev_pos     = Vec2(0,0)
        self.stoped_count = STOP_TIME
        self.stoped       = False

        self.max_x        = 0

    def mutate(self, prob, stdDev=MUT_AMMOUNT):
        """Aplica mutação ao agente
        """
        self.brain.mutation(prob=prob, stdDev=stdDev)

    def setPos(self, x, y):
        """Atualiza a posição do agente, bem como seu progresso máximo
        """
        self.curr_pos.x = x
        self.curr_pos.y = y

        if x > self.max_x:
            self.max_x = x
    
    def is_stoped(self):
        """Verifica se o agente está parado, ou se passou muito tempo sem
        obter progresso
        """
        if self.stoped:
            return True
        
        if self.prev_pos.x == self.curr_pos.x:
            self.stoped_count -= 1
        else:
            self.stoped_count = STOP_TIME
            self.prev_pos.set(self.curr_pos.x, self.curr_pos.y)
        
        if self.stoped_count < 0:
            self.stoped = True
        
        return self.stoped
        
    def setScore(self):
        """Calcula o score do agente
        """
        self.score = (self.points / 100) + self.max_x + 1
    
    def save(self, filename):
        """Salva o agente em um arquivo .pkl
        """
        with open(filename, 'wb') as save_net:
            # print("Agent saved in '{0}'".format(filename))
            pickle.dump(self, save_net)

    @staticmethod
    def load(filename):
        """Carrega um agente de um arquivo .pkl
        """
        try:
            with open(filename, 'rb') as load_net:
                # print("Agent loaded from '{0}'".format(filename))
                return pickle.load(load_net)
        except:
            # print("There is no agent saved in '{0}'".format(filename))
            return None

    def reset(self):
        """Reseta o agente para iniciar o treinamento
        """
        self.curr_pos     = Vec2(0,0)
        self.prev_pos     = Vec2(0,0)
        self.max_x        = 0
        self.stoped_count = STOP_TIME
        self.stoped       = False


    def copy(self):
        """Faz uma cópia do agente
        """
        nAgent = Agent([1,1])
        nAgent.brain = self.brain.copy()
        nAgent.points = self.points
        nAgent.score = self.score
        nAgent.fitness = self.fitness
        nAgent.generation = self.generation+1
        return nAgent



    # métodos para fins de ordenação
    def __eq__(self, other):
        return self.score == other.score
    
    def __lt__(self, other):
        return self.score < other.score    