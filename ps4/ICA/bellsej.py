### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np
import time

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def play(vec):
    sd.play(vec,Fs)
    time.sleep(8)
    sd.stop()

def sigmod_derivate(X):
    g=1/(1+np.exp(-X))
    return 1-2*g

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    p = np.random.permutation(M)
    trainData=X[p,:];
    for a in anneal:
        for i in range(M):
            W+=a*(sigmod_derivate(np.dot(W,trainData[[i],:].T)).dot(trainData[[i],:])+np.linalg.inv(W.T))
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)
    ######### Your code here ##########
    S=np.dot(W,X.T).T
    ##################################
    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])

if __name__ == '__main__':
    main()
