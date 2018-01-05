#Copyright (c) 2017 ChenyChen

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.0

import numpy as np
import operator

class ID3(object):
    def treeGrowth(self,dataSet, features):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:  # no more features
            return self.classify(classList)

        bestFeat = self.findBestSplit(dataSet)  # bestFeat is the index of best feature
        bestFeatLabel = features[bestFeat]
        myTree = {bestFeatLabel: {}}
        featValues = [example[bestFeat] for example in dataSet]
        uniqueFeatValues = set(featValues)
        del (features[bestFeat])
        for values in uniqueFeatValues:
            subDataSet = self.splitDataSet(dataSet, bestFeat, values)
            myTree[bestFeatLabel][values] = self.treeGrowth(subDataSet, features)
        return myTree

    def classify(self,classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def findBestSplit(self,dataset):
        numFeatures = len(dataset[0]) - 1
        baseEntropy = self.calcShannonEnt(dataset)
        bestInfoGain = 0.0
        bestFeat = -1
        for i in range(numFeatures):
            featValues = [example[i] for example in dataset]
            uniqueFeatValues = set(featValues)
            newEntropy = 0.0
            for val in uniqueFeatValues:
                subDataSet = self.splitDataSet(dataset, i, val)
                prob = len(subDataSet) / float(len(dataset))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            if (baseEntropy - newEntropy) > bestInfoGain:
                bestInfoGain = baseEntropy - newEntropy
                bestFeat = i
        return bestFeat

    def splitDataSet(self,dataset, feat, values):
        retDataSet = []
        for featVec in dataset:
            if featVec[feat] == values:
                reducedFeatVec = featVec[:feat]
                reducedFeatVec.extend(featVec[feat + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def calcShannonEnt(self,dataset):
        numEntries = len(dataset)
        labelCounts = {}
        for featVec in dataset:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0

        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            if prob != 0:
                shannonEnt -= prob * np.log2(prob)
        return shannonEnt

    def predict(self,tree, newObject):
        while isinstance(tree, dict):
            key = tree.keys()[0]
            tree = tree[key][newObject[key]]
        return tree