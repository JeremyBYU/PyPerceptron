
# I am purposefully keeps things extremely simple
import random
from tkinter import N

class Perceptron():
    def __init__(self, x_dim=2, learning_rate=0.1):
        self.weights = [random.random() for i in range(x_dim)]
        self.bias = random.random()
        self.learning_rate = learning_rate

    def fit(self, X, Y, epochs=1):
        n_points = len(X)
        for epoch_number in range(epochs):
            for point, true_y in zip(X, Y):
                predicted_y = self.predict(point)
                error = true_y - predicted_y
                self.update_weights(point, error)
                
    def update_weights(self, point, error):

        new_weights = []
        for w, x in zip(self.weights, point):
            w = w + self.learning_rate * error * x
            new_weights.append(w)

        self.weights = new_weights
        self.bias = self.bias + self.learning_rate * error
        


    
    def predict(X):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
