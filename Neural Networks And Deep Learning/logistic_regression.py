import numpy as np
import matplotlib as plt

def initialize_parameters(dim):
    '''
    Arguments:
    dim -- dimension for w

    Returns:
    w -- shape (dim, 1)
    b -- initialized to 0
    '''
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def preprocess_data(train_set_x_orig, test_set_x_orig):
    '''
    Preprocessing data by flattening the matrices and 
    centering and standardizing the data

    Returns dictionary containing training and testing data
    '''
    # Flatten the datasets
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    
    # Calculating means
    train_mean = np.mean(train_set_x_flatten, axis=0, keepdims=True)
    test_mean = np.mean(test_set_x_flatten, axis=0, keepdims=True)

    # Calculating standard deviations
    train_std = np.std(train_set_x_flatten, axis=0, keepdims=True)
    test_std = np.std(test_set_x_flatten, axis=0, keepdims=True)

    # Centering and Standardizing data
    train_set_x = (train_set_x_flatten - train_mean) / train_std
    test_set_x = (test_set_x_flatten - test_mean) / test_std

    return ({
        'train': train_set_x,
        'test': test_set_x
    })

def sigmoid(z):
    return (1 + np.exp(-z))**-1

def propagate(w, b, X, Y):
    '''
    Returns the gradients dw, db and the cost function's value
    for inputed values of w, b, X, and Y
    '''
    # Computing Activation
    # Broadcasting happens here when w.T is multiplied with X 
    # Broadcasting also happens when b is added to np.dot(w.T, X)
    A = sigmoid(np.dot(w.T, X) + b)
    
    # m is the number of data points in X
    m = X.shape[1]

    # Computing the cost function
    cost = np.sum(np.dot(Y,np.log(A)) + np.dot((1-Y), np.log(1-A))) / m

    # Computing the gradients
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A - Y) / m

    return {'dw': dw, 'db': db}, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    '''
    Returns the parameters w, b, gradients dw and db as well as a list 
    of cost values indexed by the number of iterations
    '''
    # Initializing return values
    costs = []
    dw = 0
    db = 0

    # Propagating and optimizing num_iterations times
    for _ in range(num_iterations):
        cost, grads = propagate(w, b, X, Y)
        costs.append(cost)
        dw = grads['dw']
        db = grads['db']
        w -= learning_rate * dw
        b -= learning_rate * dw

    # Returning costs and gradients
    return {'w': w, 'b': b}, {'dw': dw, 'db': db}, costs

def predict(w, b, X):
    ''' 
    Returns a list Y_predicted containing the predicted Y values
    '''
    # Computing activation values
    A = sigmoid(np.dot(w.T, X) + b)

    # Converting activation values to predicted values of 1 or 0
    Y_predicted = np.array([0 if a < 0.5 else 1 for a in A])

    return Y_predicted

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    '''
    This function combines all our other functions into a Logistic Regression Model

    It returns information about the model as a dictionary
    '''
    # Initializing parameters w and b
    w, b = initialize_parameters(X_train.shape[0])

    # Gradient Descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    Y_prediction_train = predict(parameters['w'], parameters['b'], X_train)
    Y_prediction_test = predict(parameters['w'], parameters['b'], X_test)

    print("Training Accuracy:", format(np.mean(np.sum(np.abs(Y_train - Y_prediction_train) * 100)), ".f2"))
    print("Testing Accuracy:", format(np.mean(np.sum(np.abs(Y_test - Y_prediction_test) * 100)), ".f2"))

    info = {
        'Y_prediction_test': Y_prediction_test,
        'Y_prediction_train': Y_prediction_train,
        'w': parameters['w'],
        'b': parameters['b'],
        'learning_rate': learning_rate,
        'num_iterations': num_iterations
        'test_error': np.mean(np.sum(np.abs(Y_train - Y_prediction_train) * 100)).
        'train_error': np.mean(np.sum(np.abs(Y_test - Y_prediction_test) * 100))
    }
    return info