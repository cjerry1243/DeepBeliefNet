import numpy as np
import numpy.matlib

def RBM(maxepoch,batchdata,numhid):
    epsilonw = 0.1  # % Learning rate for weights
    epsilonvb = 0.1  # % Learning rate for biases of visible units
    epsilonhb = 0.1  # % Learning rate for biases of hidden units
    weightcost = 0.0002
    initialmomentum = 0.5
    finalmomentum = 0.9

    [numcases, numdims, numbatches] = batchdata.shape

    # Initializing symmetric weights and biases.
    vishid = 0.1 * np.random.randn(numdims, numhid)
    hidbiases = np.zeros((1, numhid))
    visbiases = np.zeros((1, numdims))

    poshidprobs = np.zeros((numcases, numhid))
    neghidprobs = np.zeros((numcases, numhid))
    posprods = np.zeros((numdims, numhid))
    negprods = np.zeros((numdims, numhid))
    vishidinc = np.zeros((numdims, numhid))
    hidbiasinc = np.zeros((1, numhid))
    visbiasinc = np.zeros((1, numdims))
    batchposhidprobs = np.zeros((numcases, numhid, numbatches))

    for epoch in range(1, maxepoch + 1):
        print 'epoch ', epoch
        errsum = 0
        for batch in range(numbatches):
            print 'epoch ', epoch, ' batch ', batch + 1

            # %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            data = batchdata[:, :, batch]
            poshidprobs = 1. / (1 + np.exp(-np.dot(data, vishid) - np.matlib.repmat(hidbiases, numcases, 1)))
            batchposhidprobs[:, :, batch] = poshidprobs
            posprods = np.dot(data.T, poshidprobs)
            poshidact = sum(poshidprobs)
            posvisact = sum(data)

            # %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            poshidstates = poshidprobs  # %> rand(numcases,numhid)

            # %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            negdata = 1. / (1 + np.exp(-np.dot(poshidstates, vishid.T) - np.matlib.repmat(visbiases, numcases, 1)))
            neghidprobs = 1. / (1 + np.exp(-np.dot(negdata, vishid) - np.matlib.repmat(hidbiases, numcases, 1)))
            negprods = np.dot(negdata.T, neghidprobs)
            neghidact = sum(neghidprobs)
            negvisact = sum(negdata)

            # %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            err = sum(sum((data - negdata) ** 2))
            errsum = err + errsum

            if epoch > 5:
                momentum = finalmomentum
            else:
                momentum = initialmomentum

                # %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            vishidinc = momentum * vishidinc + epsilonw * ((posprods - negprods) / numcases - weightcost * vishid)
            visbiasinc = momentum * visbiasinc + (epsilonvb / numcases) * (posvisact - negvisact)
            hidbiasinc = momentum * hidbiasinc + (epsilonhb / numcases) * (poshidact - neghidact)

            vishid = vishid + vishidinc
            visbiases = visbiases + visbiasinc
            hidbiases = hidbiases + hidbiasinc
        # %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print 'epoch', epoch, 'error', errsum

    return vishid,hidbiases,batchposhidprobs

maxepoch = 10
numhid = 300
numpen = 150
numpen2 = 150

print 'Pretraining a deep autoencoder.'
print 'The Science paper used 50 epochs. This uses', maxepoch


######## makebatches for RBM
from Makebatches import batchdata

[numcases, numdims, numbatches] = batchdata.shape
########### RBM 1
print 'Pretraining Layer 1 with RBM:', numdims, '-', numhid
[vishid,hidbiases,batchposhidprobs] = RBM(maxepoch,batchdata,numhid)
w1 = np.vstack((vishid, hidbiases))
########### RBM 2
print 'Pretraining Layer 2 with RBM:', numhid, '-', numpen
batchdata = batchposhidprobs
numhid = numpen
[vishid,hidbiases,batchposhidprobs] = RBM(maxepoch,batchdata,numhid)
w2 = np.vstack((vishid, hidbiases))
########### RBM 3
print 'Pretraining Layer 3 with RBM:', numpen, '-', numpen2
batchdata = batchposhidprobs
numhid = numpen2
[vishid,hidbiases,batchposhidprobs] = RBM(maxepoch,batchdata,numhid)
w3 = np.vstack((vishid, hidbiases))
