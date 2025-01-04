import numpy as np
import pandas as pd

def z_score_normalization_train(x):
    mean = np.mean(x, axis =0)
    sigma  = np.std(x, axis=0)
    
    x_norm = (x-mean)/sigma
    
    return x_norm, mean , sigma

def z_score_normalization_test(x,mean,sigma):
    x_norm = (x-mean)/sigma
    
    return x_norm

def predictor(x,w,b):
    y_hat = np.dot(x,w) + b
    return y_hat

def compute_cost(x, y, w, b): 
    # number of training examples
    m = x.shape[0] 
    
    total_cost = 0
    cost = np.zeros(m)
    
    yhat = np.dot(x,w) + b
    cost = np.square(yhat - y)
    total_cost = np.sum(cost)
    total_cost = total_cost / (2*m)
        

    return total_cost       # which is J(w,b)



def compute_gradient(x, y, w, b): 
    
    # Number of training examples
    m = x.shape[0]
    
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    
    yhat = np.dot(x,w) + b
    dj_db = np.sum(yhat - y) / m

    yhat = yhat.reshape(m,)
    res = (yhat - y) 
    dummy_dj_dw = x*res.reshape(-1,1)
    dj_dw = (np.sum(dummy_dj_dw, axis=0))/(m)
        
    return dj_dw, dj_db



def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []  
    w_history = []  

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w_in, b_in)

        b_in -= alpha * dj_db  
        w_in -= alpha * dj_dw  
        
        # Calculate the cost function
        cost = cost_function(X, y, w_in, b_in)
        J_history.append(cost)

        # Log weights periodically (every 10% of iterations or at the last step)
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            w_history.append(w_in.copy())
            print(f"Iteration {i}: Cost {cost:.2f}")

    return w_in, b_in, J_history, w_history
