# python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
linear perceptron function parameters: xdata, y, w, alpha value
xdata represents the data on the x axis
y represents the data on the y axis
w is the vector that represents the decision boundary
alpha value is the training rate (default to 1 in this case)
'''


def linear_perceptron(x, y, w, alpha=1):
    # check if the algorithm is still running
    running = True
    # goes through all the data
    i = 0
    # iterations of the w vector
    iter = 0

    x = x.to_numpy()

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for data in range(len(x)):
        if y[data] == 1:
            x1.append(x[data][1])
            y1.append(x[data][2])
        else:
            x2.append(x[data][1])
            y2.append(x[data][2])

    # while the algorithm is still training
    while running:
        # return a 1D array of x
        temp = x[i]

        # prediction value
        yhat = 1 if sum(temp*w) >= 0 else 0

        if y[i] < yhat:
            print("Updating w. Old w = ")
            print(w)
            print("Using observation", i, ", x = ")
            print(temp)
            w = w - alpha*temp
            print("New w = ")
            print(w)
            i = 0
            iter += 1
        elif y[i] > yhat:
            print("Updating w. Old w = ")
            print(w)
            print("Using observation", i, ", x = ")
            print(temp)
            w = w + alpha*temp
            print("New w = ")
            print(w)
            i = 0
            iter += 1
        else:
            i += 1
            # if the algorithm has gone through all the datasets
            if i >= len(y):
                running = False
                print("Training completed. Final w = ")
                print(w)
                print("Total number of iterations: ", iter)
                userX = input("What is your Annual Income in $10000: ")
                userY = input("What is your Credit History in Months: ")
                plt.scatter(x1, y1, color='green', label='Good')
                plt.scatter(x2, y2, color='red', label='Bad')
                x_line = np.linspace(0, 20, 100)
                y_line = -w[1]/w[2] * x_line + -w[0]/w[2]
                plt.plot(x_line, y_line, linestyle='-',
                         color='black', label='Line')
                plt.scatter(float(userX), float(userY),
                            color='blue', label='You')
                plt.xlabel('Annual Income ($10,000)')
                plt.ylabel('Credit History in Months')
                plt.legend()
                plt.grid(True)
                plt.show()


url = 'https://raw.githubusercontent.com/garvinec/Credit-Score-Predictor/main/data.csv'
df_credit = pd.read_csv(url, index_col=0)

credit_x = df_credit.iloc[:, :3]
credit_y = df_credit.iloc[:, 3].values.flatten()

linear_perceptron(x=credit_x, y=credit_y, w=np.array([0, 1, -1]))
