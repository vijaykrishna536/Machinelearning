import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train digitrecog.csv')
#print train.describe()
test = pd.read_csv('test digitrecog.csv')
#print train.describe()

traina = train.values
y = traina[:, 0]
x = traina[:, 1:1786]
length = len(y)
#imagedata = x.reshape((43, 43))
'''
images_and_labels = list(zip(imagedata, y))
for index, [imagedata, y] in enumerate(images_and_labels[:2]):
    print "index : ", index, "image : \n ", imagedata, "label : ", y
    plt.subplot(2, 2, index+1)
    plt.axis('on')
    plt.imshow(imagedata, cmap=plt.cm.gray_r, interpolation='nearest')  # interpolation shows the clear or soft boundary
    plt.title('Training: %i' % y)

'''
print "No of train data", len(y)


testa = test.values

algo = svm.SVC(gamma=0.0001)
algo.fit(y[:(2*length)//3], x[:(2*length)//3])

expected = y[length//3:]
predicted = algo.predict(x[length//3:])

matrix = confusion_matrix(expected, predicted)
print matrix