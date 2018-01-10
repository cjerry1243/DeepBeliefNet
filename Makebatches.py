import numpy as np
import numpy.matlib
from Labels import trainlabels,testlabels
from Make_mfccdata import trainingdatapy,testingdatapy

pydatas0 =pydatas1 =pydatas2 =pydatas3 =pydatas4=pydatas5 =pydatas6 =pydatas7 =pydatas8 =pydatas9 = np.array(([]))

pydatas0.shape =pydatas1.shape =pydatas2.shape =pydatas3.shape=pydatas4.shape =pydatas5.shape =pydatas6.shape =pydatas7.shape =pydatas8.shape =pydatas9.shape = (0,trainingdatapy.shape[1])

for i in range(len(trainlabels)):
    if trainlabels[i] == 0:
        pydatas0 = np.vstack((pydatas0,trainingdatapy[i,:]))
    if trainlabels[i] == 1:
        pydatas1 = np.vstack((pydatas1,trainingdatapy[i,:]))
    if trainlabels[i] == 2:
        pydatas2 = np.vstack((pydatas2,trainingdatapy[i,:]))
    if trainlabels[i] == 3:
        pydatas3 = np.vstack((pydatas3,trainingdatapy[i,:]))
    if trainlabels[i] == 4:
        pydatas4 = np.vstack((pydatas4,trainingdatapy[i,:]))
    if trainlabels[i] == 5:
        pydatas5 = np.vstack((pydatas5,trainingdatapy[i,:]))
    if trainlabels[i] == 6:
        pydatas6 = np.vstack((pydatas6,trainingdatapy[i,:]))
    if trainlabels[i] == 7:
        pydatas7 = np.vstack((pydatas7,trainingdatapy[i,:]))
    if trainlabels[i] == 8:
        pydatas8 = np.vstack((pydatas8,trainingdatapy[i,:]))
    if trainlabels[i] == 9:
        pydatas9 = np.vstack((pydatas9,trainingdatapy[i,:]))

for j in range(len(testlabels)):
    if testlabels[j] == 0:
        pydatas0 = np.vstack((pydatas0,testingdatapy[j,:]))
    if testlabels[j] == 1:
        pydatas1 = np.vstack((pydatas1,testingdatapy[j,:]))
    if testlabels[j] == 2:
        pydatas2 = np.vstack((pydatas2,testingdatapy[j,:]))
    if testlabels[j] == 3:
        pydatas3 = np.vstack((pydatas3,testingdatapy[j,:]))
    if testlabels[j] == 4:
        pydatas4 = np.vstack((pydatas4,testingdatapy[j,:]))
    if testlabels[j] == 5:
        pydatas5 = np.vstack((pydatas5,testingdatapy[j,:]))
    if testlabels[j] == 6:
        pydatas6 = np.vstack((pydatas6,testingdatapy[j,:]))
    if testlabels[j] == 7:
        pydatas7 = np.vstack((pydatas7,testingdatapy[j,:]))
    if testlabels[j] == 8:
        pydatas8 = np.vstack((pydatas8,testingdatapy[j,:]))
    if testlabels[j] == 9:
        pydatas9 = np.vstack((pydatas9,testingdatapy[j,:]))

digitdata = np.array(([]))
digitdata.shape = (0,trainingdatapy.shape[1])

digitdata = np.vstack((digitdata,pydatas0[18:58,:],pydatas1[18:58,:],pydatas2[18:58,:],pydatas3[18:58,:],pydatas4[18:58,:],pydatas5[18:58,:],pydatas6[18:58,:],pydatas7[18:58,:],pydatas8[18:58,:],pydatas9[18:58,:]))
targets =                    np.matlib.repmat([1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],40,1)
targets = np.vstack((targets,np.matlib.repmat([0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0],40,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1],40,1)))

digitdata = (digitdata-digitdata.mean(axis=0))/(digitdata.std(axis=0)+0.0001)

totnum = digitdata.shape[0]
print 'Size of training dataset =',totnum

randomorder = np.random.permutation(totnum)

numbatches=totnum/10
numdims = digitdata.shape[1]
batchsize = 10
batchdata = np.zeros((batchsize, numdims, numbatches))
batchtargets = np.zeros((batchsize, 10, numbatches))

for b in range(numbatches):
    batchdata[:,:, b] = digitdata[randomorder[b * batchsize:(b+1) * batchsize],:]
    batchtargets[:,:, b] = targets[randomorder[b * batchsize:(b+1) * batchsize],:]

############# test

digitdata = np.array(([]))
digitdata.shape = (0,testingdatapy.shape[1])

digitdata = np.vstack((digitdata,pydatas0[8:18,:],pydatas1[8:18,:],pydatas2[8:18,:],pydatas3[8:18,:],pydatas4[8:18,:],pydatas5[8:18,:],pydatas6[8:18,:],pydatas7[8:18,:],pydatas8[8:18,:],pydatas9[8:18,:]))
targets =                    np.matlib.repmat([1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],10,1)
targets = np.vstack((targets,np.matlib.repmat([0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0],10,1)))
targets = np.vstack((targets,np.matlib.repmat([0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1],10,1)))

digitdata = (digitdata-digitdata.mean(axis=0))/(digitdata.std(axis=0)+0.0001)

totnum = digitdata.shape[0]
print 'Size of test dataset =',totnum

testrandomorder = np.random.permutation(totnum)

numbatches=totnum/10
numdims = digitdata.shape[1]
batchsize = 10
testbatchdata = np.zeros((batchsize, numdims, numbatches))
testbatchtargets = np.zeros((batchsize, 10, numbatches))

for b in range(numbatches):
    testbatchdata[:,:, b] = digitdata[testrandomorder[b * batchsize:(b+1) * batchsize],:]
    testbatchtargets[:,:, b] = targets[testrandomorder[b * batchsize:(b+1) * batchsize],:]
