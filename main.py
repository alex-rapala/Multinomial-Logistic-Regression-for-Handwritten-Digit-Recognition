import numpy as np
import pandas as pd

def load_data(path):
    """
    Loads handwritten digit data from a CSV file and returns the feature matrix and one-hot encoded class matrix.

    The feature matrix X is scaled so that its smallest element is 0 and largest element is 1. The class matrix Y
    is one-hot encoded with columns corresponding to classes in ascending order.

    :param path: the absolute path to the data file
    :return: a tuple (X, Y) where X is the feature matrix and Y is the one-hot encoded class matrix
    """
    dataFrame = pd.read_csv(path, sep=";")
    dataArray = dataFrame.to_numpy()
    yVector = dataArray[:, 0]
    xMatrix = dataArray[:, 1:]
    numObs, numFeats = xMatrix.shape
    minValue = xMatrix.min()
    maxValue = xMatrix.max()
    X = (xMatrix - minValue) / (maxValue - minValue)
    uniqueClasses = np.unique(yVector)
    numClasses = len(uniqueClasses)
    Y = np.zeros((numObs,numClasses))
    for n in range(numObs):
        number = yVector[n]
        for i in range(numClasses):
            if uniqueClasses[i] == number:
                Y[n,i] = 1
                break
    return X, Y

def image_data(X, i):
    """
    Prepares observation i from X for visualization as an image.

    :param X: the feature matrix of shape (m, p)
    :param i: the zero-based index of the observation to visualize
    :return: a square matrix suitable for image display
    """
    xRow = X[i] 
    length = int(np.sqrt(len(xRow)))
    imageMatrix = xRow.reshape(length,length)
    return 1 - imageMatrix

def grad_llh(theta, X, Y):
    """
    Computes the gradient of the log-likelihood function.

    :param theta: the parameter vector of shape ((d-1)(p+1),)
    :param X: the feature matrix of shape (m, p)
    :param Y: the one-hot encoded class matrix of shape (m, d)
    :return: the gradient vector of shape ((d-1)(p+1),)
    """
    #start by finding all dimensions
    numObs, numFeats = X.shape
    numClasses = Y.shape[1]

    #split the vs and ws
    reshapedMatrix = theta.reshape(numClasses - 1, numFeats +1)
    vVector = reshapedMatrix[:, 0]
    wVector = reshapedMatrix[:, 1:]

    #calculations
    expMatrix = np.exp(vVector + X @ wVector.T)
    normConstVector = 1 + np.sum(expMatrix, axis=1)
    QMatrix = expMatrix / normConstVector.reshape(numObs,1)
    qLastClass = (1 / normConstVector).reshape(numObs,1)
    fullQMatrix = np.column_stack([QMatrix, qLastClass])

    #get gradients
    differenceMatrix = Y - fullQMatrix
    vGradient = np.sum(differenceMatrix[:, :numClasses-1], axis=0)
    wGradient = differenceMatrix[:, :numClasses-1].T @ X

    parameterMatrix = np.column_stack([vGradient, wGradient])
    gradient = parameterMatrix.flatten()
    return gradient

def bisection(grad, L, delta):
    """
    Finds a zero of a function's derivative using the bisection method.

    :param grad: the derivative of the function
    :param L: the starting point where grad(L) < 0
    :param delta: the tolerance for convergence
    :return: the point M where grad(M) is approximately zero
    """
    U = L + delta
    while grad(U) < 0:
        U = L + 2 * (U-L)
    while U - L > delta:
        M = (L + U) / 2
        if grad(M) < 0:
            L = M
        elif grad(M) > 0:
            U = M
        else:
            L = M
            U = M
    return (L + U) / 2

def line_search_BFGS(grad, x0, Binv0, delta, epsilon):
    """
    Performs line search optimization using the BFGS method.

    The algorithm iterates until the gradient norm falls below epsilon. Step length is determined using the
    bisection method with tolerance delta.

    :param grad: the gradient function of the objective to minimize
    :param x0: the starting point
    :param Binv0: the initial inverse Hessian approximation
    :param delta: the bisection method tolerance
    :param epsilon: the gradient norm tolerance for convergence
    :return: a list of iterates visited during optimization
    """
    numParameters = len(x0)
    currentX = x0
    currentInvB = Binv0
    iterateList = [x0]
    currentGradient = grad(currentX)
    while np.linalg.norm(currentGradient) > epsilon:

        searchDirection = -1 * currentInvB @ currentGradient

        def phiDerivative(alpha):
            newX = currentX + alpha * searchDirection
            newGradient = grad(newX)
            return newGradient @ searchDirection

        stepLength = bisection(phiDerivative, 0, delta)
        nextX = currentX + stepLength * searchDirection
        nextGradient = grad(nextX)
        posChange = nextX - currentX
        gradChange = nextGradient - currentGradient
        constant = np.dot(gradChange, posChange)
        nextInvB = (np.identity(numParameters) - np.outer(posChange, gradChange)/constant) @ currentInvB @ (np.identity(numParameters) - np.outer(gradChange, posChange)/constant) + np.outer(posChange, posChange)/constant
        
        currentX = nextX
        currentInvB = nextInvB
        currentGradient = nextGradient
        iterateList.append(currentX)

    return iterateList

def mlogit(X, Y):
    """
    Fits a multinomial logistic regression model by maximizing the regularized log-likelihood.

    Uses line search with BFGS direction to optimize. The regularization parameter is set to 0.01 * m.

    :param X: the feature matrix of shape (m, p)
    :param Y: the one-hot encoded class matrix of shape (m, d)
    :return: the optimal parameter vector theta
    """
    numObs, numFeats = X.shape
    numClasses = Y.shape[1]
    numParam = (numClasses - 1) * (numFeats + 1)
    rho = 0.01 * numObs
    delta = 10**-2
    epsilon = 0.1 * np.sqrt(numObs * numParam)
    Binv0 = (numClasses / numObs) * np.identity(numParam)
    x0 = np.zeros(numParam)

    def gradRegLLH(theta):
        return -grad_llh(theta, X, Y) + rho * theta

    iterateList = line_search_BFGS(gradRegLLH, x0, Binv0, delta, epsilon)
    return iterateList[-1]

def mlogit_accuracy(theta, X, Y):
    """
    Computes the accuracy of multinomial logistic regression predictions.

    Accuracy is computed as the proportion of correctly classified observations.

    :param theta: the parameter vector of shape ((d-1)(p+1),)
    :param X: the feature matrix of shape (m, p)
    :param Y: the one-hot encoded class matrix of shape (m, d)
    :return: the accuracy as a float between 0 and 1
    """
    numObs, numFeats = X.shape
    numClasses = Y.shape[1]
    numParam = (numClasses - 1) * (numFeats + 1)
    reshapedMatrix = theta.reshape(numClasses - 1, numFeats + 1)
    vVector = reshapedMatrix[:, 0]
    wVector = reshapedMatrix[:, 1:]
    expMatrix = np.exp(vVector + X @ wVector.T)
    normConstVector = 1 + np.sum(expMatrix, axis=1)
    QMatrix = expMatrix / normConstVector.reshape(numObs,1)
    qLastClass = (1 / normConstVector).reshape(numObs,1)
    fullQMatrix = np.column_stack([QMatrix, qLastClass])
    predictedClasses = np.argmax(fullQMatrix, axis = 1) 
    trueClasses = np.argmax(Y, axis = 1)
    numCorrect = np.sum(predictedClasses == trueClasses)
    return numCorrect / numObs

def train_test_split(X, Y, beta, seed):
    """
    Splits data into training and testing sets.

    Randomly allocates observations to training and testing sets based on the fraction beta. Uses the given seed
    for reproducibility.

    :param X: the feature matrix of shape (m, p)
    :param Y: the one-hot encoded class matrix of shape (m, d)
    :param beta: the fraction of observations for training data
    :param seed: the random seed for reproducibility
    :return: a tuple trainX, trainY, testX, testY
    """
    numObs, numFeats = X.shape

    rng = np.random.default_rng(seed)
    sampleSize = int(np.round(beta * numObs))

    trainIndices = rng.choice(numObs, sampleSize, replace=False)
    trainX = X[trainIndices, :]
    trainY = Y[trainIndices, :]

    testIndices = np.setdiff1d(np.arange(numObs), trainIndices)
    testX = X[testIndices, :]
    testY = Y[testIndices, :]

    return trainX, trainY, testX, testY

def split_mlogit_accuracies(X, Y):
    """
    Trains a multinomial logistic regression model and computes in-sample and out-of-sample accuracies.

    Splits the data with 70% training and 30% testing using seed 123456. Trains the model on training data
    and evaluates accuracy on both sets.

    :param X: the feature matrix of shape (m, p)
    :param Y: the one-hot encoded class matrix of shape (m, d)
    :return: a tuple (inSampleAccuracy, outSampleAccuracy)
    """
    beta = 0.7
    seed = 123456
    trainX, trainY, testX, testY = train_test_split(X, Y, beta, seed)
    theta = mlogit(trainX, trainY)
    inSampleAccuracy = mlogit_accuracy(theta, trainX, trainY)
    outSampleAccuracy = mlogit_accuracy(theta, testX, testY)
    return inSampleAccuracy, outSampleAccuracy