import numpy as np
import h5py
import copy
from random import randint


MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

input_size = 28*28 
output = 10 
hidden_layer = 100

bias1 = np.zeros((hidden_layer, 1))
bias2 = np.zeros((output, 1))
W = np.random.randn(hidden_layer,input_size) / np.sqrt(input_size)
C = np.random.randn(output,hidden_layer) / np.sqrt(hidden_layer)

def convert_y(y):
    arr = np.zeros((output,1))
    arr[y] = 1
    return arr

def softmax_function(z):
    ZZ = np.exp(z - max(z))/np.sum(np.exp(z - max(z)))
    return ZZ

def gradient_tanh(z):
    return (1-np.power(np.tanh(z),2))

'''
def Relu(z):
    return np.maximum(z,0)

def gradient_Relu(z):
    return np.where(z>0,1,0)
'''

LR = .01
num_epochs = 20

for epochs in range(num_epochs):

    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001

    total_correct = 0

    for n in range(len(x_train)):
        n_random = randint(0,len(x_train)-1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        x = np.reshape(x, (784,1))

        #forward propagation
        Z = np.dot(W, x) + bias1
        H = np.tanh(Z)
        U = np.dot(C, H) + bias2
        rho = softmax_function(U) 
        predicted_value = np.argmax(rho)

        if (predicted_value == y):
            total_correct += 1

        #backward propagation 
        diff_U = rho - convert_y(y)
        diff_bias2 = diff_U
        diff_C = np.matmul(diff_U, H.transpose())
        delta = np.matmul(C.transpose(),diff_U) 
        sigma_dash = gradient_tanh(H)
        diff_bias1 = np.multiply(delta, sigma_dash)
        diff_W = np.matmul(diff_bias1, x.transpose())

        #parameter updation
        C = C - LR*diff_C
        bias1 = bias1 - LR*diff_bias1
        bias2 = bias2 - LR*diff_bias2
        W = W - LR*diff_W

    print("Training accuracy for epoch {} : {}".format(epochs+1, total_correct/np.float(len(x_train))))

#test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x = np.reshape(x, (input_size,1))

    Z = W.dot(x) + bias1
    H = np.tanh(Z)
    U = np.dot(C, H) + bias2
    rho = softmax_function(U)
    predicted_value = np.argmax(rho)
    
    if (predicted_value == y):
        total_correct += 1


print("Test accuracy : {}".format(total_correct/np.float(len(x_test))))







