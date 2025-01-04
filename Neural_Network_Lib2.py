import numpy as np
import pandas as pd

def z_score_normalization_train(x):
    mean = np.mean(x, axis=0)
    sigma = np.std(x, axis=0) + 1e-8  # Added epsilon for numerical stability
    x_norm = (x - mean) / sigma
    return x_norm, mean, sigma

def z_score_normalization_test(x, mean, sigma):
    x_norm = (x - mean) / (sigma + 1e-8)  # Added epsilon for numerical stability
    return x_norm

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_derivative(a):
    return a * (1 - a)

def linear(x, w, b):
    return np.dot(x, w.T) + b

def relu_activation(a):
    return np.maximum(0, a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def dense(a_in, w, b, activation_function):
    z = linear(a_in, w, b)
    return activation_function(z)

def forward_propagation(x, w1, b1, w2, b2, w3, b3, w4, b4):
    a1 = dense(x, w1, b1, relu_activation)
    a2 = dense(a1, w2, b2, relu_activation)
    a3 = dense(a2, w3, b3, relu_activation)
    a4 = dense(a3, w4, b4, relu_activation)
    return a1, a2, a3, a4

def binary_output_probability(a4, w5, b5):
    return dense(a4, w5, b5, sigmoid)

def class_output(a4, w5, b5):
    return dense(a4, w5, b5, softmax)

def backward_propagation_sigmoid(x, y, a1, a2, a3, a4, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, output, learning_rate=0.001):
    output_error = y.reshape(-1, 1) - output
    output_delta = output_error * sigmoid_derivative(output)

    a4_error = np.dot(output_delta, w5)
    a4_delta = a4_error * (a4 > 0)

    a3_error = np.dot(a4_delta, w4)
    a3_delta = a3_error * (a3 > 0)

    a2_error = np.dot(a3_delta, w3)
    a2_delta = a2_error * (a2 > 0)

    a1_error = np.dot(a2_delta, w2)
    a1_delta = a1_error * (a1 > 0)

    w1 -= learning_rate * np.dot(x.T, a1_delta).T
    w2 -= learning_rate * np.dot(a1.T, a2_delta).T
    w3 -= learning_rate * np.dot(a2.T, a3_delta).T
    w4 -= learning_rate * np.dot(a3.T, a4_delta).T
    w5 -= learning_rate * np.dot(a4.T, output_delta).T

    b1 -= learning_rate * np.sum(a1_delta, axis=0, keepdims=True)
    b2 -= learning_rate * np.sum(a2_delta, axis=0, keepdims=True)
    b3 -= learning_rate * np.sum(a3_delta, axis=0, keepdims=True)
    b4 -= learning_rate * np.sum(a4_delta, axis=0, keepdims=True)
    b5 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5

def backward_softmax(x, y_one_hot, a1, a2, a3, a4, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, output, learning_rate=0.01):
    output_error = y_one_hot - output
    output_delta = output_error

    a4_error = np.dot(output_delta, w5)
    a4_delta = a4_error * (a4 > 0)

    a3_error = np.dot(a4_delta, w4)
    a3_delta = a3_error * (a3 > 0)

    a2_error = np.dot(a3_delta, w3)
    a2_delta = a2_error * (a2 > 0)

    a1_error = np.dot(a2_delta, w2)
    a1_delta = a1_error * (a1 > 0)

    w1 -= learning_rate * np.dot(x.T, a1_delta).T
    w2 -= learning_rate * np.dot(a1.T, a2_delta).T
    w3 -= learning_rate * np.dot(a2.T, a3_delta).T
    w4 -= learning_rate * np.dot(a3.T, a4_delta).T
    w5 -= learning_rate * np.dot(a4.T, output_delta).T

    b1 -= learning_rate * np.sum(a1_delta, axis=0, keepdims=True)
    b2 -= learning_rate * np.sum(a2_delta, axis=0, keepdims=True)
    b3 -= learning_rate * np.sum(a3_delta, axis=0, keepdims=True)
    b4 -= learning_rate * np.sum(a4_delta, axis=0, keepdims=True)
    b5 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5

def predict_softmax(a_out):
    return np.argmax(a_out, axis=1) + 1

def predict_binary(a_out):
    return (a_out >= 0.5).astype(int)

def accuracy(result, y):
    acc = (np.sum(result == y.reshape(-1, 1)) / len(y)) * 100
    print(f"The accuracy is {acc:.2f}%")

def y_one_hot_converter(y, num_classes=10):
    y_one_hot = np.zeros((y.shape[0], num_classes), dtype=int)
    for i, j in enumerate(y):
        y_one_hot[i][j - 1] = 1
    return y_one_hot
