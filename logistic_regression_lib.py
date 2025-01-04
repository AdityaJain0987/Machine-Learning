import numpy as np
import pandas as pd

def z(x,w,b):
    z = np.dot(x,w) + b
    return(z)
    
def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, *argv):
    
    m, n = X.shape
    w = w.reshape(n,1)
    
    
    
    z = np.dot(X,w) + b
    f = 1/(1 + np.exp(-z))
    f = f.reshape(m,)
    epsilon = 1e-15
    f = np.clip(f, epsilon, 1 - epsilon)

    loss = -np.dot(y,np.log(f)) - np.dot((1 -y),np.log(1-f))
    total_cost = np.sum(loss)/m

    return total_cost

def z_score_normalization_train(x):
    mean = np.mean(x, axis =0)
    sigma  = np.std(x, axis=0)

    x_norm = (x-mean)/sigma
    
    return x_norm, mean , sigma

def z_score_normalization_test(x,mean,sigma):
    x_norm = (x-mean)/sigma
    
    return x_norm


def compute_gradient(X, y, w, b, *argv): 
    
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    w = w.reshape(n,1)
    
    z = np.dot(X,w) + b
    f = 1/(1 + np.exp(-z))
    f = f.reshape(m,)
    res = f-y
    dj_db = np.sum(res)/m
    dj_dw = X*(res).reshape(-1, 1)
    dj_dw = np.sum(dj_dw, axis =0)/m
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []  
    w_history = []  

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in)

        w_in -= alpha * dj_dw
        b_in -= alpha * dj_db

        cost = cost_function(X, y, w_in, b_in)
        J_history.append(cost)

        # Log weights periodically (every 10% of iterations or at the last step)
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            w_history.append(w_in.copy())
            print(f"Iteration {i}: Cost {cost:.2f}")

    return w_in, b_in, J_history, w_history


def predict(x, w, b): 
    
    # number of training examples
    m, n = x.shape   
    p = np.zeros(m)
    
    z = np.dot(x,w) + b
    f = 1/(1 + np.exp(-z))
    
    for i in range(m):
        if(f[i]>= .5):
            f[i] = 1
        else:
            f[i] = 0
        p[i] = f[i]
    
    return p







