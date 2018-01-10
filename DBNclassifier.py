import numpy as np
import scipy.io as sio
import numpy.matlib

def FEED_FORWARD(data,N,w1,w2,w3,w_class):
    data = np.column_stack((data, np.ones((N, 1))));
    w1probs = 1. / (1 + np.exp(-np.dot(data, w1)));
    w1probs = np.column_stack((w1probs, np.ones((N, 1))));
    w2probs = 1. / (1 + np.exp(-np.dot(w1probs, w2)));
    w2probs = np.column_stack((w2probs, np.ones((N, 1))));
    w3probs = 1. / (1 + np.exp(-np.dot(w2probs, w3)));
    w3probs = np.column_stack((w3probs, np.ones((N, 1))));
    targetout = np.exp(np.dot(w3probs, w_class));
    targetout = targetout / np.matlib.repmat(sum(targetout.T), 10, 1).T;
    return targetout

def CG_CLASSIFY(w1, w2, w3, w_class, XX, target):
    N = XX.shape[0];
    XX = np.column_stack((XX, np.ones((N, 1))));
    w1probs = 1. / (1 + np.exp(-np.dot(XX, w1)));
    w1probs = np.column_stack((w1probs, np.ones((N, 1))));

    w2probs = 1. / (1 + np.exp(-np.dot(w1probs, w2)));
    w2probs = np.column_stack((w2probs, np.ones((N, 1))));

    w3probs = 1. / (1 + np.exp(-np.dot(w2probs, w3)));
    w3probs = np.column_stack((w3probs, np.ones((N, 1))));

    targetout = np.exp(np.dot(w3probs, w_class));
    targetout = targetout / np.matlib.repmat(sum(targetout.T), 10, 1).T;

    # f = -sum(sum(target* np.log(targetout)+ (1-target)* np.log(1-targetout)));

    IO = targetout - target;
    Ix_class = IO;
    dw_class = np.dot(w3probs.T, Ix_class);

    Ix3 = np.dot(Ix_class, w_class.T) * w3probs * (1 - w3probs);
    Ix3 = Ix3[:, :-1];
    dw3 = np.dot(w2probs.T, Ix3);

    Ix2 = np.dot(Ix3, w3.T) * w2probs * (1 - w2probs);
    Ix2 = Ix2[:, :-1];
    dw2 = np.dot(w1probs.T, Ix2);

    Ix1 = np.dot(Ix2, w2.T) * w1probs * (1 - w1probs);
    Ix1 = Ix1[:, :-1];
    dw1 = np.dot(XX.T, Ix1);

    return dw1, dw2, dw3, dw_class;

def RBM(maxepoch,restart,batchdata,numhid):
    epsilonw = 0.1;  # % Learning rate for weights
    epsilonvb = 0.1;  # % Learning rate for biases of visible units
    epsilonhb = 0.1;  # % Learning rate for biases of hidden units
    weightcost = 0.0002;
    initialmomentum = 0.5;
    finalmomentum = 0.9;

    [numcases, numdims, numbatches] = batchdata.shape;

    if restart == 1:
        restart = 0;
        epoch = 1;

        # % Initializing symmetric weights and biases.
        vishid = 0.1 * np.random.randn(numdims, numhid);
        hidbiases = np.zeros((1, numhid));
        visbiases = np.zeros((1, numdims));

        poshidprobs = np.zeros((numcases, numhid));
        neghidprobs = np.zeros((numcases, numhid));
        posprods = np.zeros((numdims, numhid));
        negprods = np.zeros((numdims, numhid));
        vishidinc = np.zeros((numdims, numhid));
        hidbiasinc = np.zeros((1, numhid));
        visbiasinc = np.zeros((1, numdims));
        batchposhidprobs = np.zeros((numcases, numhid, numbatches));
    # be aware of matrix index
    for epoch in range(epoch, maxepoch + 1):
        print 'epoch ', epoch;
        errsum = 0;
        for batch in range(numbatches):
            print 'epoch ', epoch, ' batch ', batch + 1;

            # %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata[:, :, batch];
            poshidprobs = 1. / (1 + np.exp(-np.dot(data, vishid) - np.matlib.repmat(hidbiases, numcases, 1)));
            batchposhidprobs[:, :, batch] = poshidprobs;
            posprods = np.dot(data.T, poshidprobs);
            poshidact = sum(poshidprobs);
            posvisact = sum(data);

            # %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs;  # %> rand(numcases,numhid);

            # %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1. / (1 + np.exp(-np.dot(poshidstates, vishid.T) - np.matlib.repmat(visbiases, numcases, 1)));
            neghidprobs = 1. / (1 + np.exp(-np.dot(negdata, vishid) - np.matlib.repmat(hidbiases, numcases, 1)));
            negprods = np.dot(negdata.T, neghidprobs);
            neghidact = sum(neghidprobs);
            negvisact = sum(negdata);

            # %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            err = sum(sum((data - negdata) ** 2));
            errsum = err + errsum;

            if epoch > 5:
                momentum = finalmomentum;
            else:
                momentum = initialmomentum;

                # %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods) / numcases - weightcost * vishid);
            visbiasinc = momentum * visbiasinc + (epsilonvb / numcases) * (posvisact - negvisact);
            hidbiasinc = momentum * hidbiasinc + (epsilonhb / numcases) * (poshidact - neghidact);

            vishid = vishid + vishidinc;
            visbiases = visbiases + visbiasinc;
            hidbiases = hidbiases + hidbiasinc;
        # %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print 'epoch', epoch, 'error', errsum;

    return vishid,hidbiases,batchposhidprobs

maxepoch = 10;
numhid = 300;
numpen = 150;
numpen2 = 150;

print 'Pretraining a deep autoencoder.';
print 'The Science paper used 50 epochs. This uses', maxepoch;


######## makebatches for RBM
from Makebatches import batchdata

[numcases, numdims, numbatches] = batchdata.shape;
########### RBM 1
print 'Pretraining Layer 1 with RBM:', numdims, '-', numhid;
restart = 1;
[vishid,hidbiases,batchposhidprobs] = RBM(maxepoch,restart,batchdata,numhid);
w1 = np.vstack((vishid, hidbiases));
########### RBM 2
print 'Pretraining Layer 2 with RBM:', numhid, '-', numpen;
batchdata = batchposhidprobs;
numhid = numpen;
restart = 1;
[vishid,hidbiases,batchposhidprobs] = RBM(maxepoch,restart,batchdata,numhid);
w2 = np.vstack((vishid, hidbiases));
########### RBM 3
print 'Pretraining Layer 3 with RBM:', numpen, '-', numpen2;
batchdata = batchposhidprobs;
numhid = numpen2;
restart = 1;
[vishid,hidbiases,batchposhidprobs] = RBM(maxepoch,restart,batchdata,numhid);
w3 = np.vstack((vishid, hidbiases));


########################### Backpropclassify
maxepoch = 1000;
print 'Training discriminative model by minimizing cross entropy error.';
print '40 batches of 10 cases each.';

######### makebatches for backpropogation
from Makebatches import batchdata,batchtargets,testbatchdata,testbatchtargets

w_class = 0.1 * np.random.randn(w3.shape[1] + 1, 10);

train_err = np.array(([]));
train_crerr = np.array(([]));
test_err = np.array(([]));
test_crerr = np.array(([]));

learnrate = 0.002;

for epoch in range(maxepoch):
    err_cr = 0;
    match = 0;
    [numcases, numdims, numbatches] = batchdata.shape;
    N = numcases;
    for batch in range(numbatches):
        data = batchdata[:, :, batch];
        target = batchtargets[:, :, batch];
        targetout = FEED_FORWARD(data,N,w1,w2,w3,w_class);

        J = np.argmax(targetout, axis=1);
        J1 = np.argmax(target, axis=1);
        for i in range(10):
            if J[i] == J1[i]:
                match = match + 1;
        err_cr = err_cr - sum(sum(target * np.log(targetout)));

    train_err = np.append(train_err, numcases * numbatches - match);
    train_crerr = np.append(train_crerr, err_cr / numbatches);


    err_cr = 0;
    match = 0;

    [testnumcases, testnumdims, testnumbatches] = testbatchdata.shape;
    N = testnumcases;
    for batch in range(testnumbatches):
        data = testbatchdata[:, :, batch];
        target = testbatchtargets[:, :, batch];
        targetout = FEED_FORWARD(data, N, w1, w2, w3, w_class);

        J = np.argmax(targetout, axis=1);
        J1 = np.argmax(target, axis=1);
        for i in range(10):
            if J[i] == J1[i]:
                match = match + 1;
        err_cr = err_cr - sum(sum(target * np.log(targetout)));

    test_err = np.append(test_err, testnumcases*testnumbatches - match);
    test_crerr = np.append(test_crerr, err_cr/testnumbatches);
    '''
    if np.mod(epoch+1, 10) == 0:
        plt.figure(1);
        plt.plot(train_err);
        drawnow()
        plt.figure(2);
        plt.plot(test_err);
        plt.show();
    '''
    print 'Before epoch ', epoch + 1;
    print 'Train misclassified: ', train_err[epoch], '(from ', numcases * numbatches, ').';
    print 'Test misclassified: ', test_err[epoch], '(from ', testnumcases * testnumbatches, ').';

    ########### Combine 2 batches to 1 large batch
    tt = 0;
    for batch in range(numbatches/4):
        print 'epoch ', epoch + 1, 'batch ', batch + 1;

        tt = tt + 1;
        data = batchdata[:, :, (tt-1)*4];
        targets = batchtargets[:, :, (tt-1)*4];
        for kk in range(4-1):
            data = np.vstack((data,batchdata[:,:, (tt-1)*4 + kk+1]));
            targets = np.vstack((targets,batchtargets[:,:, (tt-1)*4 + kk+1]));
        '''''''''
        if epoch > 100:
            learnrate = 0.001;
        '''''''''
        [dw1, dw2, dw3, dw_class] = CG_CLASSIFY(w1, w2, w3, w_class, data, targets);
        w1 = w1 - learnrate * dw1;
        w2 = w2 - learnrate * dw2;
        w3 = w3 - learnrate * dw3;
        w_class = w_class - learnrate * dw_class;
