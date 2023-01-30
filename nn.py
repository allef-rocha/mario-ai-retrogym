import numpy as np
from itertools import product

W_MIN = 0.01
W_MAX = 2.00

# Valores padrão 
MUT_RATE    = 0.05  # chance de mutação
MUT_AMMOUNT = 0.10  # desvio padrão de mutação

class ActivationFunction:
	def __init__(self, func, dfunc):
		self.func = func
		self.dfunc = dfunc

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def dsigmoid(z):
	sigm = sigmoid(z)
	return sigm * (1 - sigm)

def tanh(z):
	return np.tanh(z)

def dtanh(z):
	t = np.tanh(z)
	return 1 - np.power(t,2)

class NeuralNetwork:
    def __init__(self, topology:list):

        # Numero de neuronios por layer
        self.shape = topology.copy()

        # Pesos inicializados com zero
        num_weights = len(topology)-1
        self.weights = [np.zeros((topology[i]+1, topology[i+1])) for i in range(num_weights)]
        
        # Função de ativação
        self.actv = ActivationFunction(sigmoid, dsigmoid)

        # Mutação inicial
        self.mutation(prob=0.5,stdDev=0.5)

    def print(self):
        """ Imprime as matrizes peso de forma legível
        """
        with np.printoptions(precision=4, suppress=True, formatter={'float': '{: 0.4f}'.format}, linewidth=100):
            for i, w in enumerate(self.weights):
                print(f'W_{i}:')
                print(w)
                print()

    def predict(self, input):
        """ FeedFoward do input através das matrizes peso
        """
        # curr -> layer atual
        curr = np.array(input, dtype=np.float)
        for w in self.weights:
            # É adcionado um nó extra, sempre igual a 1 (bias)
            curr_with_bias = np.hstack((np.ones((curr.shape[0], 1)), curr))
            
            # Produto matricial entre o layer atual (com bias) e a matriz peso
            prod = np.matmul(curr_with_bias, w)
            
            # É aplicada a função de ativação
            curr = self.actv.func(prod)
        return curr

    @staticmethod
    def crossover(A, B):
        """ Recebe duas NeuralNetworks, A e B, e faz um crossover de cada 
            matriz peso. A matriz é particionada, herdando a primeira parte
            do parente A, e o restante do parente B

                      partition = 3
                          |
                          v

                a   a   a | b   b
                a   a   a | b   b
                a   a   a | b   b
                a   a   a | b   b
        """
        child = A.copy()
        for idx, w in enumerate(child.weights):
            B_w = B.weights[idx]
            row, col = w.shape
            partition = np.int(np.random.random() * col) + 1
            w[:, partition:] = B_w[:, partition:]
        return child


    def mutation(self, prob=MUT_RATE, stdDev=MUT_AMMOUNT):
        """ Cada link presente na rede neural tem uma chance (prob) de
            sofrer uma mutação, sendo esta extraída de uma função normal
            (media = 0, desvio padrão = stdDev)
        """
        for w in self.weights:
            rows, cols = w.shape
            for i, j in product(range(rows), range(cols)):
                if np.random.rand() < prob:
                    w[i, j] += np.random.randn() * stdDev

                    # Caso o valor, após a mutação, exceda os limites, é ajustado
                    absW = abs(w[i,j])
                    if absW > W_MAX:
                        w[i,j] = W_MAX * np.sign(w[i,j])
                    elif absW < W_MIN:
                        w[i,j] = 0.0

    def copy(self):
        """ Retorna uma cópia da rede neural
        """
        nNN = NeuralNetwork(self.shape)
        for i,w in enumerate(self.weights):
            nNN.weights[i] = w.copy()
        return nNN

