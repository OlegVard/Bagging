import math
import random


class Network:
    def __init__(self, n_ins, n_hidden, n_hidden_neurons, n_out, data):
        self.data = data
        self.data_iter = 0
        self.inputs = n_ins
        self.hidden = n_hidden
        self.hidden_neurons = n_hidden_neurons
        self.outs = n_out
        self.weights = []
        self.create_network()

    @staticmethod
    def generate_weights(neuron_count):
        weights = []
        for i in range(neuron_count):
            weights.append(random.random())
        return weights

    def create_network(self):
        self.weights.append(self.generate_weights(self.inputs))
        for _ in range(self.hidden):
            self.weights.append(self.generate_weights(self.hidden_neurons))
        self.weights.append(self.generate_weights(self.outs))

    def feed_forward(self):
        data = self.data[self.data_iter]
        self.data_iter += 1
        results = []
        for i in range(len(self.weights)):
            prom_results = []
            for j in range(len(self.weights[i])):
                if i == 0:
                    prom_results.append(self.activate(data[j] * self.weights[i][j]))
                else:
                    neuron_mean = sum(results[i - 1]) * self.weights[i][j]
                    prom_results.append(self.activate(neuron_mean))
            results.append(prom_results.copy())
        self.back_propagation(results, data[-2:])

    def back_propagation(self, results, reference):
        learning_rate = 0.5
        error = reference[0] - results[self.hidden + 1][0] + reference[1] - results[self.hidden][1]
        correction = error / (results[self.hidden + 1][0] + results[self.hidden][1])
        print(error)
        print(reference[0], results[self.hidden + 1][0], reference[1], results[self.hidden][1])
        for i in range(len(self.weights) - 1, - 1, -1):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] + (1 - results[i][j]) * learning_rate * correction

    @staticmethod
    def activate(x):
        return 1 / (1 + math.exp(-x))
