
# I am purposefully keeps things extremely simple
from numpy import random
from .logger import log

random.seed(4)

class Perceptron():
    def __init__(self, x_dim=2, learning_rate=0.1):
        self.weights = [random.random() for i in range(x_dim)]
        self.bias = random.random()
        self.learning_rate = learning_rate

    def fit(self, X, Y, epochs=1):
        """This will fit the perceptron to the training data.


        Args:
            X (list): list of points (each point is a list of scalar values)
            Y (list): list of classifications (ground truth)
            epochs (int, optional): How many times to repeat training on training data. Defaults to 1.
        """        
        n_points = len(X)
        adjusted_learning_rate = self.learning_rate / n_points
        all_data = []
        for epoch_number in range(epochs):
            for point, true_y in zip(X, Y):
                predicted_y = self.predict(point)
                error = true_y - predicted_y
                self.update_weights(point, error, adjusted_learning_rate)
                log.debug(f"true_y: {true_y}, predicted_y: {predicted_y}, error: {error}")
            accuracy, y_predicted = self.accuracy(X, Y)
            log.info(f"Epoch: {epoch_number}; Accuracy: {accuracy}")
            all_data.append(dict(y_predicted=y_predicted, accuracy=accuracy, weights=self.weights, bias=self.bias))
        return all_data
                
    def update_weights(self, point, error, adjusted_learning_rate=0.1):
        """Will update the weights of the perceptron based on the error

        Args:
            point (list): a list of scalar values
            error (float): the error associated with this point
            adjusted_learning_rate (float, optional): Learning rate. Defaults to 0.1.
        """        

        new_weights = []
        for w, x in zip(self.weights, point):
            w = w + adjusted_learning_rate * error * x
            new_weights.append(w)

        self.weights = new_weights
        self.bias = self.bias + adjusted_learning_rate * error

    def activation_fn(self, y):
        """Heaviside step function

        Args:
            y (float): A class value

        Returns:
            int: 0 or 1
        """        
        return 1 if y > 0 else 0
        
    def predict(self, point):
        """Will predict the class of a point

        Args:
            point (list): a point is a list of scalar values

        Returns:
            float: a class predictions (0 or 1)
        """
        y = self.bias        
        for w, x in zip(self.weights, point):
            y += w * x # each weight is multiplied by the corresponding x
        
        y = self.activation_fn(y) # apply the activation function, heaviside step function

        return y

    def accuracy(self, X, Y):
        """Will calculate the accuracy of the Perceptron

        Args:
            X (list): list of points (each point is a list of scalar values)
            Y (list): list of classifications

        Returns:
            float: Accuracy of the Perceptron (0.0 - 1.0)
        """
        correct = 0
        y_predicted_values = []
        for point, y_true in zip(X, Y):
            y_pred = self.predict(point)
            y_predicted_values.append(y_pred)
            if y_pred == y_true:
                correct += 1
        return correct / len(Y), y_predicted_values


def main():
    pass


if __name__ == "__main__":
    main()
