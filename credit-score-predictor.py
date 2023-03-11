# python libraries
import pandas as pd
import numpy as np

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

    # while the algorithm is still training
    while running:
        # return a 1D array of x
        x = x.iloc[i, :]
        # prediction value
        yhat = 1 if sum(x*w) >= 0 else 0

        if y[i] < yhat:
            print("Updating w. Old w = ")
            print(w)
            print("Using observation", i, ", x = ")
            print(x.to_numpy())
            w = w - alpha*x
            print("New w = ")
            print (w)
            i = 0
            iter += 1
        elif y[i] > yhat:
            print("Updating w. Old w = ")
            print(w)
            print("Using observation", i, ", x = ")
            print(x.to_numpy())
            w = w + alpha*x
            print("New w = ")
            print (w)
            i = 0
            iter += 1
        else:
            i += 1
            # if the algorithm has gone through all the datasets
            if i >= len(y):
                running = False
                print ("Training completed. Final w = ")
                print (w)
                print("Total number of iterations = ")
                print (iter)

url = 'https://raw.githubusercontent.com/garvinec/Credit-Score-Predictor/main/data.csv'
df_credit = pd.read_csv(url, index_col = 0)

credit_x = df_credit.iloc[:, :3]
credit_y = df_credit.iloc[:, 3].values.flatten()

linear_perceptron(x=credit_x, y=credit_y, w=np.array([0,1,-1]))
