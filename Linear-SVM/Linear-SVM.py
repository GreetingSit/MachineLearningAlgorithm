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

class LinearSVM(object):
    def __init__(self):
        #use to initialize arguments.
        self._w=self._b=None

    def fit(self,x,y,c=1,lr=0.01,epoch=10000):
        '''
        use to fit _w and _b
        arguments:
                x:input data
                y:labels
                c:cost
                lr:learning rate
                epoch:iterations
        return:
        '''
        x,y=np.asarray(x,np.float32),np.asarray(y,np.float32)
        self._w=np.zeros(x.shape[1])
        self._b=0.
        for _ in range(epoch):
            self._w*=1-lr
            err=1-y*self.predict(x,True)
            idx=np.argmax(err)
            if err[idx]<=0:
                d=lr*c*y[idx]
                self._w+=d*x[idx]
                self._b+=d

    def predict(self,x,raw=False):
        '''
        use to predict
        argumets:
                x:sample
                raw:if True return value of y_pred else return sucess or not
        return:
                y_pred:the value of w*x+b
                sign:sucess or not
        '''
        x=np.asarray(x,np.float32)
        y_pred=x.dot(self._w)+self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)