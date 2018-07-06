import numpy as np

import matplotlib.pyplot as plt
import math
import json

NUM_INPUT=784
NUM_HIDDEN=300
NUM_OUTPUT=10
BATCH=1000

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    s=np.exp(x);
    for c_index in range(s.shape[1]):
        s[:,c_index]=np.true_divide(s[:,c_index],np.sum(s[:,c_index]))
	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s=1/(1+np.exp(-x))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    data_num=data.shape[0];
    hidden_x=np.dot(W1,np.transpose(data))+np.dot(b1,np.ones((1,data_num)))
    h=sigmoid(hidden_x)
    ### END YOUR CODE
    y_x=np.dot(W2,h)+np.dot(b2,np.ones((1,data_num)))
    y=softmax(y_x)
    cost=0;
    for i in range(data_num):
        cost=cost-np.dot(labels[i,:],np.log(y[:,i]))
    labmda=0.00005
    cost=cost/data_num+labmda*(np.sum(np.power(W1,2))+np.sum(np.power(W2,2)));
    return h, y, cost

def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    (h, y, cost) = forward_prop(data, labels, params)
    (data_num,data_col)=data.shape
    (label_num,label_col)=labels.shape
    gradW2=np.zeros((NUM_OUTPUT,NUM_HIDDEN))
    gradb2=np.zeros((NUM_OUTPUT,1))
    gradW1=np.zeros((NUM_HIDDEN,NUM_INPUT))
    gradb1=np.zeros((NUM_HIDDEN,1))
    for label_index in range(label_num):
        # compute grad_z
        grad_z=y[:,label_index].reshape(NUM_OUTPUT,1)-labels[label_index,:].reshape(NUM_OUTPUT,1)
        # compute grad_w2
        h_item=h[:,label_index].reshape(NUM_HIDDEN,1)
        gradW2+=np.dot(grad_z,np.transpose(h_item))
        gradb2+=grad_z
        #compute grad_h
        grad_h=np.dot(np.transpose(grad_z),W2).T
        #compute grad_W1
        gradW1+=np.dot(grad_h*h_item*(1-h_item),data[label_index,:].reshape(1,NUM_INPUT))
        #compute grad_b1
        gradb1+=(grad_h*h_item*(1-h_item))
    labmda = 0.00005
    gradW1=gradW1/data_num+2*labmda*W1
    gradW2=gradW2/data_num+2*labmda*W2
    gradb1=gradb1/data_num
    gradb2=gradb2/data_num
    ### END YOUR CODE
    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    params = {}

    ### YOUR CODE HERE
    W1=(np.random.rand(NUM_HIDDEN,NUM_INPUT)-0.5)*0.074/0.5
    W1=W1.reshape(NUM_HIDDEN,NUM_INPUT)
    b1=(np.random.rand(NUM_HIDDEN)-0.5)*0.074/0.5
    b1=b1.reshape(NUM_HIDDEN,1)
    W2=(np.random.rand(NUM_OUTPUT,NUM_HIDDEN)-0.5)*0.14/0.5
    W2=W2.reshape(NUM_OUTPUT,NUM_HIDDEN);
    b2=(np.random.rand(NUM_OUTPUT)-0.5)*0.14/0.5
    b2=b2.reshape(NUM_OUTPUT,1)
    params['W1']=W1
    params['b1']=b1
    params['W2']=W2
    params['b2']=b2
    iterNum=[]
    trainCost=[]
    trainAccuracy=[]
    devCost=[]
    devAccuracy=[]
    for i in range(30):
        (h, y, cost) = forward_prop(trainData, trainLabels, params)
        print('the iter count:',i,' cost:',cost)
        iterNum.append(i)
        trainCost.append(cost)
        trainAccuracy.append(compute_accuracy(y.T,trainLabels))
        (h,y,cost)=forward_prop(devData,devLabels,params)
        devCost.append(cost)
        devAccuracy.append(compute_accuracy(y.T,devLabels))
        for b_index in range(int(trainData.shape[0]/BATCH)):
            row_start=b_index*BATCH
            row_end=(b_index+1)*BATCH
            grad=backward_prop(trainData[row_start:row_end,:],trainLabels[row_start:row_end,:],params)
            params['W1']=params['W1']-learning_rate*grad['W1']
            params['W2']=params['W2']-learning_rate*grad['W2']
            params['b1']=params['b1']-learning_rate*grad['b1']
            params['b2']=params['b2']-learning_rate*grad['b2']
    ### END YOUR CODE
    plt.figure(1)
    plt.plot(iterNum,trainCost)
    plt.plot(iterNum,devCost,color="r")
    plt.figure(2)
    plt.plot(iterNum, trainAccuracy)
    plt.plot(iterNum, devAccuracy, color="r")
    plt.show()
    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output.T, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels = readData('mnist/images_train.csv', 'mnist/labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('mnist/images_test.csv', 'mnist/labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    params = nn_train(trainData, trainLabels, devData, devLabels)
    np.savez("params",W1=params['W1'],W2=params['W2'],b1=params['b1'],b2=params['b2'])
    params=np.load('params.npz')
    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print('Test accuracy: %f' % accuracy)

if __name__ == '__main__':
    main()
