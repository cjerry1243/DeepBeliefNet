import numpy as np

f=open('labels/trainlabels.txt', 'r')
trainlabels = np.array(([]))
for label in f.readlines():
    if label[0] == '0':
        trainlabels = np.append(trainlabels,0)
    if label[0] == '1':
        trainlabels = np.append(trainlabels,1)
    if label[0] == '2':
        trainlabels = np.append(trainlabels,2)
    if label[0] == '3':
        trainlabels = np.append(trainlabels,3)
    if label[0] == '4':
        trainlabels = np.append(trainlabels,4)
    if label[0] == '5':
        trainlabels = np.append(trainlabels,5)
    if label[0] == '6':
        trainlabels = np.append(trainlabels,6)
    if label[0] == '7':
        trainlabels = np.append(trainlabels,7)
    if label[0] == '8':
        trainlabels = np.append(trainlabels,8)
    if label[0] == '9':
        trainlabels = np.append(trainlabels,9)
f.close()

f=open('labels/testlabels.txt', 'r')
testlabels = np.array(([]))
for label in f.readlines():
    if label[0] == '0':
        testlabels = np.append(testlabels,0)
    if label[0] == '1':
        testlabels = np.append(testlabels,1)
    if label[0] == '2':
        testlabels = np.append(testlabels,2)
    if label[0] == '3':
        testlabels = np.append(testlabels,3)
    if label[0] == '4':
        testlabels = np.append(testlabels,4)
    if label[0] == '5':
        testlabels = np.append(testlabels,5)
    if label[0] == '6':
        testlabels = np.append(testlabels,6)
    if label[0] == '7':
        testlabels = np.append(testlabels,7)
    if label[0] == '8':
        testlabels = np.append(testlabels,8)
    if label[0] == '9':
        testlabels = np.append(testlabels,9)
f.close()
