from numpy import *
import numpy
# from sklearn.model_selection import train_test_split

iris = loadtxt('diabetes.data',delimiter=',')
iris[:,:8] = iris[:,:8]-iris[:,:8].mean(axis=0)
iris[:,:8] = iris[:,:8]
# print (iris[0:9,:])

target = zeros((shape(iris)[0],7));
indices = where(iris[:,8]==0) 
target[indices,0] = 1
indices = where(iris[:,8]==1)
target[indices,1] = 1
indices = where(iris[:,8]==2)
target[indices,2] = 1

order = range(shape(iris)[0])
random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

train = iris[::2,0:8]
traint = target[::2]
valid = iris[1::4,0:8]
validt = target[1::4]
test = iris[3::4,0:8]
testt = target[3::4]

import mlp
net = mlp.mlp(train,traint,5,outtype='logistic')

net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)
