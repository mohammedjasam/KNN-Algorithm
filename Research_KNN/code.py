###   Scripted by Mohammed Jasam   ###

import csv
import random
import math
import numpy as np
from sklearn import metrics
import operator
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
import IPython
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyPDF2 import PdfFileMerger
import os
import subprocess


count=1
l=[]
def colCount(filename):
    datafilename = 'data.csv'
    d = ','
    f=open(datafilename,'r')

    reader=csv.reader(f,delimiter=d)
    ncol=len(next(reader)) # Read first line and count columns
    f.seek(0)              # go back to beginning of file
    for row in reader:
        pass
    return(ncol)

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)
            next(lines)
            dataset = list(lines)
            cols=colCount('data.csv')
            # 5 Fold Cross Validation!!!
            i=0
            if count==1:
                for x in range(len(dataset)-1):
                    i+=1
                    for y in range(cols):
                        dataset[x][y] = float(dataset[x][y])
                    if 20<=i<100 :
                        trainingSet.append(dataset[x])
                    else:
                        testSet.append(dataset[x])
                        # print(i)
            elif count==2:
                for x in range(len(dataset)-1):
                    i+=1
                    for y in range(cols):
                        dataset[x][y] = float(dataset[x][y])
                    if 0<=i<20:
                        trainingSet.append(dataset[x])
                    elif 40<=i<100 :
                        trainingSet.append(dataset[x])
                    else:
                        testSet.append(dataset[x])
                        # print(i)
            elif count==3:
                for x in range(len(dataset)-1):
                    i+=1
                    for y in range(cols):
                        dataset[x][y] = float(dataset[x][y])
                    if 0<=i<40:
                        trainingSet.append(dataset[x])
                    elif 60<=i<100 :
                        trainingSet.append(dataset[x])
                    else:
                        testSet.append(dataset[x])
                        # print(i)
            elif count==4:
                for x in range(len(dataset)-1):
                    i+=1
                    for y in range(cols):
                        dataset[x][y] = float(dataset[x][y])
                    if 0<=i<60:
                        trainingSet.append(dataset[x])
                    elif 80<=i<100 :
                        trainingSet.append(dataset[x])
                    else:
                        testSet.append(dataset[x])
                        # print(i)
            else:
                for x in range(len(dataset)-1):
                    i+=1
                    for y in range(cols):
                        dataset[x][y] = float(dataset[x][y])
                    if 0<=i<80:
                        trainingSet.append(dataset[x])
                    else:
                        testSet.append(dataset[x])
                        # print(i)


def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
                distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
                dist = euclideanDistance(testInstance, trainingSet[x], length)
                distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
                neighbors.append(distances[x][0])
        return neighbors

def getResponse(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
                response = neighbors[x][-1]
                if response in classVotes:
                        classVotes[response] += 1
                else:
                        classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
                if testSet[x][-1] == predictions[x]:
                        correct += 1
        return (correct/float(len(testSet))) * 100.0

def main():
        # prepare data
        trainingSet=[]
        testSet=[]
        split = 0.78
        loadDataset('data.csv', split, trainingSet, testSet)
        # print ('Train set: ' + repr(len(trainingSet)))
        # print ( 'Test set: ' + repr(len(testSet)))
        # generate predictions
        predictions=[]
        k = 3
        for x in range(len(testSet)):
                neighbors = getNeighbors(trainingSet, testSet[x], k)
                result = getResponse(neighbors)
                predictions.append(result)
                # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        pre=[]
        act=[]
        # print(repr(result))
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = getResponse(neighbors)
            predictions.append(result)
            pre.append(repr(result))
            act.append(repr(testSet[x][-1]))
        # print(pre)
        pre = [float(i) for i in pre]
        pre = [int(i) for i in pre]
        # print(pre)
        act = [float(i) for i in act]
        act = [int(i) for i in act]
        # print(act)
        # print(pre)
        predict = np.array([pre])
        expect = np.array([act])

        # print(predict)
        smp_report = metrics.precision_recall_fscore_support(pre, act, beta=1, average = None)
        f_report = metrics.f1_score(pre, act, average =None)

        print(f_report)#, smp_report)
        # except:
        #     pass

        # nums = [int(x) for x in pre]
        # e = list(map(int, pre))
        # print(act)
        accuracy = getAccuracy(testSet, predictions)
        l.append(accuracy)
        # print('Accuracy: ' + repr(accuracy) + '%')

for x in range(5):
    main()
    count+=1
l=[int(x) for x in l]
with open("KNN_Accuracy.csv", "w") as fp_out:
    writer = csv.writer(fp_out, delimiter=",")
    writer.writerow(l)
# print(l)
a=0
for i in range(len(l)):
    a+=l[i]
# print(a)
# print('The average accuracy after 5 Fold Cross Validation is '+str(a/len(l))+'%')
objects=('Fold 1','Fold 2','Fold 3','Fold 4','Fold 5')
y_pos = np.arange(len(objects))
plt.bar(y_pos, l)
plt.show()

sys.exit()
