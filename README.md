# DeepBeliefNet
Pre-training a deep neural networks by rbms, and use backprpogation for spoken digit classification.


## To Start
Download all files into a folder, and unzip 2 files: trainingwav.zip, testingwav.zip.
Original wav files are in the 2 unzipped folder,including training and testing files.
Their labels are in the 'labels' folder

## Pyfiles
#### Labels.py 
reads training and testing labels, converting labels to list form.

#### Make_mfccdata.py 
reads wav files, making a 26x10=260 dimension vector for each file.

#### Makebatches.py 
collects mfcc datas and their labels, makeing a mini-batch (batchsize=10) as input for Deep Belief Networks.

#### Pretraining.py 
performs RBM process,stacking RBM to a Deep Belief Networks.

#### DBNclassifier.py 
use Backpropogation gradient descend method to classify voice of numbers.

## To Run
run 'DBNclassifier.py' to see the result.

## Result
accuracy: 0.92
