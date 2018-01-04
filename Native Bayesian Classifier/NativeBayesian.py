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

class NativeBayesian(object):
    def __init__(self,dataset):
        '''
        use to init
        arguments:
                dataset: training dataset
        '''
        self._dataset=dataset
        self._P,self._p=self.train(dataset)
        self._n=len(self._P.keys())

    def train(self):
        '''
        training
        return:
                P:probability of labels[i]
                p:probability of p(k/i)
        '''
        labellist=self._dataset[:,-1]
        labels={}
        sum=0.0
        P={}
        p={}
        for label in labellist:
            if label in labels.keys():
                labels[label]+=1.0
                sum+=1.0
            else:
                labels[label]=1.0
                sum+=1.0
        for label in labels:
            P[label]=labels[label]/sum
        for i in range(self._dataset.shape[0]-1):
            p[i]={}
            for label in labels.keys():
                sum=0.0
                for k in self._dataset.shape[1]:
                    if self._dataset[k][-1] == label:
                        if self._dataset[k][i] in p[i].keys():
                            p[i][self._dataset[k][i]]+=1.0
                            sum+=1.0
                        else:
                            p[i][self._dataset[k][i]]=1.0
                            sum+=1.0
                for k in p[i]:
                    p[i][k]/=sum

        return P, p

    def classify(self,test):
        '''
        classify
        arguments:
                test:test sample
        return:
                label:label
        '''
        ans=np.zeros(self._n)
        for i in range(self._n):
            ans[i]*=np.log2(self._p[i][test[i]])
        for i in range(self._n):
            ans[i]*=np.log2(self._P[i])
        label=np.argmax(ans)
        return label





