from doctest import debug
from timeit import repeat
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

from pyperceptron import Perceptron, log
viridis = cm.get_cmap('viridis', 12)


def get_endpoints(axes, slope, y_intercept):
    x_vals = np.array(axes.get_xlim())
    y_val_0 = y_intercept + slope * x_vals[0]
    y_val_1 = y_intercept + slope * x_vals[1]
    return x_vals, [y_val_0, y_val_1]


def main():

    # Make fake data
    X, Y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=1)


    # Plot the data
    log.info("Plotting data, color denotes class. Close figure when done (click X on window)")
    fig, ax = plt.subplots(figsize=(8, 7), nrows=1, ncols=1)
    ax.scatter(X[:,0], X[:,1], c=Y, s=50, cmap='viridis')
    plt.show()

    # Creating the perceptron class and fitting data
    log.info("Creating perceptron and fitting data")
    X_list = X.tolist() 
    Y_list = Y.tolist()
    n_epochs = 50
    perceptron = Perceptron(x_dim=2, learning_rate=0.101)
    all_data = perceptron.fit(X_list, Y_list, epochs=n_epochs)

    input("Perceptron has been trained. Press ENTER to show animated graph of training and results")


    # All the remaining code is just for plotting data
    # Creating an animation of the data
    fig, ax = plt.subplots(figsize=(17,5), nrows=1, ncols=3)
    scat_true = ax[0].scatter(X[:,0], X[:,1], c=Y, s=50, cmap='viridis')
    scat_pred = ax[1].scatter(X[:,0], X[:,1], c=Y, s=50, cmap='viridis')
    seperating_line,  = ax[1].plot([], [], '--')
    xdata, ydata = [], []
    ln, = ax[2].plot([], [], 'ro')

    ax[0].set_title("Ground Truth Data")
    ax[1].set_title("Predicted Data")
    ax[2].set_title("Accuracy")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[2].set_xlabel("# Epochs")
    ax[2].set_ylabel("Accuracy %")

    fig.tight_layout()

    def init():
        ax[2].set_xlim(0, n_epochs)
        ax[2].set_ylim(0, 100.0)


        return scat_true, scat_pred, ln, seperating_line

    def update(frame):
        accuracy = all_data[frame]['accuracy'] * 100
        weights = all_data[frame]['weights']
        bias = all_data[frame]['bias']

        # change color in scatter plot, middle axis
        prediction = all_data[frame]['y_predicted']
        scat_pred.set_array(prediction)

        # Set the seperating line, middle axis
        slope = - weights[0] / weights[1]
        y_intercept = - bias / weights[1]
        x_vals, y_vals = get_endpoints(ax[1], slope, y_intercept)
        seperating_line.set_data(x_vals, y_vals)
        
        # Set accuracy data (last axis)
        xdata.append(frame)
        ydata.append(accuracy)
        ln.set_data(xdata, ydata)

        return scat_true, scat_pred, ln, seperating_line

    # Create animation and save
    ani = FuncAnimation(fig, update, frames=range(n_epochs), init_func=init, blit=True, repeat=False)
    plt.show()
    xdata = []
    ydata = []
    ani.save('animation.gif', writer=PillowWriter(fps=5))


if __name__ == "__main__":
    main()
