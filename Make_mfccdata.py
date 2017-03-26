import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc



sheets = 10
mfcc_filter = 26

def filetest(path, words):
    (rat,sig) = wav.read(path)
    if len(sig.shape)==2:
        sig0=np.array(sig[:,0])
        sig1=np.array(sig[:,1])
    else:
        sig0=sig
    special_mfcc_feat = mfcc(sig0,rat,numcep=26)
    boundary = int(special_mfcc_feat.shape[0]/words)
    spl = int(special_mfcc_feat.shape[0]/words/sheets)
    specialmfcc = np.zeros((words, sheets*mfcc_filter))
    for w in xrange(words):
        for i in xrange(sheets):
            specialmfcc[w][mfcc_filter*i:mfcc_filter*(i+1)]=np.mean(special_mfcc_feat[boundary*w+spl*i:boundary*w+spl*(i+1)], axis=0)
    return specialmfcc

words = 1
i=0
f=open('training/trainingfiles.txt','r')
for fname in f.readlines():
    i = i + 1
    fname = fname.strip()
    specialmfcc = filetest('training/' + fname, words)
    if i ==1:
        trainingdatapy = specialmfcc
    else:
        trainingdatapy = np.vstack((trainingdatapy,specialmfcc))
    if i ==580:
        break
f.close()

i=0
f=open('testing/testingfiles.txt','r')
for fname in f.readlines():
    i = i + 1
    fname = fname.strip()
    specialmfcc = filetest('testing/' + fname, words)
    if i ==1:
        testingdatapy = specialmfcc
    else:
        testingdatapy = np.vstack((testingdatapy,specialmfcc))
    if i ==120:
        break
f.close()

