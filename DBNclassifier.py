import numpy as np
import numpy.matlib

def FEED_FORWARD(data,N,w1,w2,w3,w_class):
    data = np.column_stack((data, np.ones((N, 1))))
    w1probs = 1. / (1 + np.exp(-np.dot(data, w1)))
    w1probs = np.column_stack((w1probs, np.ones((N, 1))))
    w2probs = 1. / (1 + np.exp(-np.dot(w1probs, w2)))
    w2probs = np.column_stack((w2probs, np.ones((N, 1))))
    w3probs = 1. / (1 + np.exp(-np.dot(w2probs, w3)))
    w3probs = np.column_stack((w3probs, np.ones((N, 1))))
    targetout = np.exp(np.dot(w3probs, w_class))
    targetout = targetout / np.matlib.repmat(sum(targetout.T), 10, 1).T
    return targetout
def CG_CLASSIFY(w1, w2, w3, w_class, XX, target):
    N = XX.shape[0]
    XX = np.column_stack((XX, np.ones((N, 1))))
    w1probs = 1. / (1 + np.exp(-np.dot(XX, w1)))
    w1probs = np.column_stack((w1probs, np.ones((N, 1))))

    w2probs = 1. / (1 + np.exp(-np.dot(w1probs, w2)))
    w2probs = np.column_stack((w2probs, np.ones((N, 1))))

    w3probs = 1. / (1 + np.exp(-np.dot(w2probs, w3)))
    w3probs = np.column_stack((w3probs, np.ones((N, 1))))

    targetout = np.exp(np.dot(w3probs, w_class))
    targetout = targetout / np.matlib.repmat(sum(targetout.T), 10, 1).T

    # f = -sum(sum(target* np.log(targetout)+ (1-target)* np.log(1-targetout)))

    IO = targetout - target
    Ix_class = IO
    dw_class = np.dot(w3probs.T, Ix_class)

    Ix3 = np.dot(Ix_class, w_class.T) * w3probs * (1 - w3probs)
    Ix3 = Ix3[:, :-1]
    dw3 = np.dot(w2probs.T, Ix3)

    Ix2 = np.dot(Ix3, w3.T) * w2probs * (1 - w2probs)
    Ix2 = Ix2[:, :-1]
    dw2 = np.dot(w1probs.T, Ix2)

    Ix1 = np.dot(Ix2, w2.T) * w1probs * (1 - w1probs)
    Ix1 = Ix1[:, :-1]
    dw1 = np.dot(XX.T, Ix1)

    return dw1, dw2, dw3, dw_class

########################### Pretraining
from Pretraining import w1,w2,w3

########################### Backpropclassify
maxepoch = 1000
print 'Training discriminative model by minimizing cross entropy error.'
print '40 batches of 10 cases each.'

########### Makebatches for Backpropogation
from Makebatches import batchdata,batchtargets,testbatchdata,testbatchtargets

w_class = 0.1 * np.random.randn(w3.shape[1] + 1, 10)

train_err = np.array(([]))
train_crerr = np.array(([]))
test_err = np.array(([]))
test_crerr = np.array(([]))

learnrate = 0.002

for epoch in range(maxepoch):
    err_cr = 0
    match = 0
    [numcases, numdims, numbatches] = batchdata.shape
    N = numcases
    for batch in range(numbatches):
        data = batchdata[:, :, batch]
        target = batchtargets[:, :, batch]
        targetout = FEED_FORWARD(data,N,w1,w2,w3,w_class)

        J = np.argmax(targetout, axis=1)
        J1 = np.argmax(target, axis=1)
        for i in range(10):
            if J[i] == J1[i]:
                match = match + 1
        err_cr = err_cr - sum(sum(target * np.log(targetout)))

    train_err = np.append(train_err, numcases * numbatches - match)
    train_crerr = np.append(train_crerr, err_cr / numbatches)


    err_cr = 0
    match = 0

    [testnumcases, testnumdims, testnumbatches] = testbatchdata.shape
    N = testnumcases
    for batch in range(testnumbatches):
        data = testbatchdata[:, :, batch]
        target = testbatchtargets[:, :, batch]
        targetout = FEED_FORWARD(data, N, w1, w2, w3, w_class)

        J = np.argmax(targetout, axis=1)
        J1 = np.argmax(target, axis=1)
        for i in range(10):
            if J[i] == J1[i]:
                match = match + 1
        err_cr = err_cr - sum(sum(target * np.log(targetout)))

    test_err = np.append(test_err, testnumcases*testnumbatches - match)
    test_crerr = np.append(test_crerr, err_cr/testnumbatches)
    '''
    if np.mod(epoch+1, 10) == 0:
        plt.figure(1)
        plt.plot(train_err)
        drawnow()
        plt.figure(2)
        plt.plot(test_err)
        plt.show()
    '''
    print 'Before epoch ', epoch + 1
    print 'Train misclassified: ', train_err[epoch], '(from ', numcases * numbatches, ').'
    print 'Test misclassified: ', test_err[epoch], '(from ', testnumcases * testnumbatches, ').'

    ########### Combine 4 batches to 1 large batch
    tt = 0
    for batch in range(numbatches/4):
        print 'epoch ', epoch + 1, 'batch ', batch + 1

        tt = tt + 1
        data = batchdata[:, :, (tt-1)*4]
        targets = batchtargets[:, :, (tt-1)*4]
        for kk in range(4-1):
            data = np.vstack((data,batchdata[:,:, (tt-1)*4 + kk+1]))
            targets = np.vstack((targets,batchtargets[:,:, (tt-1)*4 + kk+1]))
        '''''''''
        if epoch > 100:
            learnrate = 0.001
        '''''''''

        ################ Update Weights
        [dw1, dw2, dw3, dw_class] = CG_CLASSIFY(w1, w2, w3, w_class, data, targets)
        w1 = w1 - learnrate * dw1
        w2 = w2 - learnrate * dw2
        w3 = w3 - learnrate * dw3
        w_class = w_class - learnrate * dw_class
