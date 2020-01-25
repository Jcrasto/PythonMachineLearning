import sys,os
import pandas
import matplotlib.pyplot as plt
import numpy

#%%
def main():
    df = pandas.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    y = df.iloc[0:100, 4].values
    y = numpy.where(y == "Iris-Setosa", -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    
    plt.scatter(X[:50,0], X[:50,1], color='red', marker = 'x', label = 'versicolor')
    plt.ylabel('sepal length')
    plt.xlabel('petal length')
    plt.legend(loc = 'upper left')
    plt.show()

    from utils.perceptron import Perceptron
    ppn = Perceptron(eta = 0.1, n_iter=10)
    ppn.fit(X,y)
    plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,marker ='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()

#%%
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    main()
