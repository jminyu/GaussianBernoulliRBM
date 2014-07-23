'''
Created on 2014. 7. 2.

@author: Eric Schmits
'''
import sys
import os
import cPickle
import matplotlib.pylab as plt
import matplotlib.cm as cm
#import Image

import numpy as np
import h5py as h5
from numpy import *
from scipy import *
from numpy.core import *
from numpy.random import *
from scipy import stats, mgrid, c_, reshape, random, rot90
from scipy import ndimage
import pylab as p

sys.path.append('../code')
sys.path.append('C:\Python27\Lib\site-packages')

from deepbelief import DBN, GaussianRBM, SemiRBM, RBM
from imageprocessing import Image

def genData():
    c1 = 0.5
    r1 = 0.4
    r2 = 0.3
    Ndat = 1000
    # generate enough data to filter
    N = 20* Ndat
    X = array(random_sample(N))
    Y = array(random_sample(N))
    X1 = X[(X-c1)*(X-c1) + (Y-c1)*(Y-c1) < r1*r1]
    Y1 = Y[(X-c1)*(X-c1) + (Y-c1)*(Y-c1) < r1*r1]
    X2 = X1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
    Y2 = Y1[(X1-c1)*(X1-c1) + (Y1-c1)*(Y1-c1) > r2*r2]
    X3 = X2[ abs(X2-Y2)>0.05 ]
    Y3 = Y2[ abs(X2-Y2)>0.05 ]
    #X3 = X2[ X2-Y2>0.15 ]
    #Y3 = Y2[ X2-Y2>0.15]
    X4=zeros( Ndat, dtype=float32)
    Y4=zeros( Ndat, dtype=float32)
    for i in xrange(Ndat):
        if (X3[i]-Y3[i]) >0.05:
            X4[i] = X3[i] + 0.08
            Y4[i] = Y3[i] + 0.18
        else:
            X4[i] = X3[i] - 0.08
            Y4[i] = Y3[i] - 0.18
    print "X", size(X3[0:Ndat]), "Y", size(Y3)
    return vstack((X4[0:Ndat],Y4[0:Ndat])), vstack((array(random_sample(Ndat)),array(random_sample(Ndat))))

def load_data_BS(dataset):
    # Load the dataset
    #f = gzip.open(dataset, 'rb')
    f = open(dataset, 'r')    
    train_set= cPickle.load(f)
    print train_set.shape
    f.close()
    print train_set.shape
    print train_set.shape[0]-5
    noOfExamples = train_set.shape[0]-5
    train_set_x = train_set[:noOfExamples,:]
    #train_set_x = train_set[243:257,:] 
    test_set_x = train_set[noOfExamples:,:]
    train_set_x0 = train_set_x[:,::3]
    train_set_x1 = train_set_x[:,1::3]
    train_set_x2 = train_set_x[:,2::3]
    #print test_set_x,train_set_x
    return train_set_x0.T,train_set_x1.T,train_set_x2.T, test_set_x.T

def subtract(x,y,sigma,scalesigma):
    #scalesigma = 11
    x = x.flatten()
    y = y.flatten()
    sigma = sigma.flatten()
    a = (x+sigma*scalesigma)>y
    b = (x-sigma*scalesigma)<y
    return a*b*1
    



def f_measure(fg,gt):
    fg_pixgt = where(gt == 1)
    bg_pixgt = where(gt == 0)

    fg_pixfg = where(fg == 1)
    bg_pixfg = where(fg == 0)


    tp = (where(gt[fg_pixgt] == fg[fg_pixgt]))[0]
    fp = (where(gt[fg_pixgt] != fg[fg_pixgt]))[0]
    tn = (where(gt[bg_pixgt] == fg[bg_pixgt]))[0]
    fn = (where(gt[bg_pixgt] != fg[bg_pixgt]))[0]


    DR = len(tp) * 1.0 / ( len(tp) + len(fn))
    percision = len(tp) * 1.0 / (len(tp) + len(fp))

    F = 2 * DR * percision / (DR + percision)
    return F


def main(argv):
    # load preprocessed data samples
    print 'loading data...\t',
    #data_train, data_test = genData() #load('../data/vanhateren.npz')
    DIR1 = "../data/changedetection/camerajitter/traffic"
    data_train_0,data_train_1,data_train_2, data_test = load_data_BS(DIR1 + "/orignal_color.pkl")
    img = Image.open(DIR1 + "/gt001537.png")
    
    x_, y_ = (asarray(img)[:,:]).shape #determine procesing image size
    print "Doie : " , x_,y_
    
    groundtruth = (((asarray(img)[:,:])/255.0 > 0.5) * 1).flatten()
    
    #groundtruth = (((asarray(img)[:,:])/255.0 > 0.5) * 1).flatten()
    print '[DONE]'
    

    # remove DC component (first component)
    # data_train = data['train'][1:, :]
    # data_test = data['test'][1:, :]

    # create 1st layer
    dbn = DBN(GaussianRBM(num_visibles=data_train_0.shape[0], num_hiddens=20))
    dbn1 = DBN(GaussianRBM(num_visibles=data_train_1.shape[0], num_hiddens=20))
    dbn2 = DBN(GaussianRBM(num_visibles=data_train_2.shape[0], num_hiddens=20))
    
    dbn[0].learning_rate = 0.001
    dbn1[0].learning_rate = 0.001
    dbn2[0].learning_rate = 0.001

   

    # train 1st layer
    print 'training...\t',
    dbn.train(data_train_0, num_epochs=1, batch_size=1,shuffle=False)
    dbn1.train(data_train_1, num_epochs=1, batch_size=1,shuffle=False)
    dbn2.train(data_train_2, num_epochs=1, batch_size=1,shuffle=False)
    print '[DONE]'

    
    data_test_0 = ((data_test.T)[:,::3]).T
    data_test_1 = ((data_test.T)[:,1::3]).T
    data_test_2 = ((data_test.T)[:,2::3]).T
    
    # global runtime
    # sFilename = "data\\weightlogs"
    # f = h5.File(sFileName+".hdf5")
    # f.create_group(runtime)
    # group.create_dataset('parameters',data=par)

    #storeWeights(category,"data\\weightlogs", dbn[0].W, dbn[0].vsigma,dbn1[0].W, dbn1[0].vsigma,dbn2[0].W, dbn2[0].vsigma )

    Ndat = 20 #data_test_0.shape[1]
    Nsteps = 5
    # evaluate 1st layer
    print 'evaluating 1...\t',
    dataout = zeros(x_*y_)
    # #datasub = zeros(x_*y_)
    for point in xrange(Ndat):
        #X = asmatrix(data_test_0[:,point]).T
        X = asmatrix(data_test_0[:,-1]).T
        #dataout = vstack((dataout,X.flatten()))
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = dbn[0].forward(X) # self.activ(1)
            X = dbn[0].backward(Y,X)
        #print "S hsape:", X.shape
        #dataout = vstack((dataout,X.flatten()))
        dataout = vstack((dataout,subtract(asarray(X),data_test_0[:,-1],asarray(dbn[0].vsigma),point+1)))
    
    print 'evaluating 2...\t',
    dataout1 = zeros(x_*y_)
    # #datasub = zeros(x_*y_)
    for point in xrange(Ndat):
        #X = asmatrix(data_test_1[:,point]).T
        X = asmatrix(data_test_1[:,-1]).T
        #dataout1 = vstack((dataout1,X.flatten()))
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = dbn1[0].forward(X) # self.activ(1)
            X = dbn1[0].backward(Y,X)
        #print "S hsape:", X.shape
        #dataout1 = vstack((dataout1,X.flatten()))
        dataout1 = vstack((dataout1,subtract(asarray(X),data_test_1[:,-1],asarray(dbn1[0].vsigma),point+1)))
    
    
    print 'evaluating 3...\t',
    dataout2 = zeros(x_*y_)
    # #datasub = zeros(x_*y_)
    for point in xrange(Ndat):
        #X = asmatrix(data_test_2[:,point]).T
        X = asmatrix(data_test_2[:,-1]).T
        #dataout2 = vstack((dataout2,X.flatten()))
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = dbn2[0].forward(X) # self.activ(1)
            X = dbn2[0].backward(Y,X)
        #print "S hsape:", X.shape
        #dataout2 = vstack((dataout2,X.flatten()))
        dataout2 = vstack((dataout2,subtract(asarray(X),data_test_2[:,-1],asarray(dbn2[0].vsigma),point+1)))
    
    # plt.imshow((reshape(data_test[::3,5],(x_,y_))), cmap = cm.Greys_r, interpolation ="nearest")
    # plt.axis('off')     
    # plt.show()
    fmeasureV = zeros(Ndat)
    plt.figure(2)
    for i in range(Ndat):
        plt.subplot(5,5,i+1)
        d = multiply(asarray(dataout[i+1,:]),asarray(dataout1[i+1,:]),asarray(dataout2[i+1,:]))
        d = mod(d+1,2)
        fmeasureV[i] = f_measure(d,groundtruth)
        print "Image Example Fmeaure: ",i," : ", fmeasureV[i] * 100
        # d[0::3] = asarray(dataout[i+1,:])
        # d[1::3] = asarray(dataout1[i+1,:])
        # d[2::3] = asarray(dataout2[i+1,:])
        # d[:,:,0] = (reshape(asarray(dataout[i+1,:]),(x_,y_)))
        # d[:,:,1] = (reshape(asarray(dataout1[i+1,:]),(x_,y_)))
        # d[:,:,2] = (reshape(asarray(dataout2[i+1,:]),(x_,y_)))
        # img_s = Image.fromarray(asarray(reshape(d,(x_,y_))*255,dtype="uint8"))
        # img_s.save("C:\work\\backgdSubt\dataset\datasets\change detection\\baseline\\baseline\office\grbm" + "\\" + str(i) + ".bmp")
        plt.imshow(reshape(d,(x_,y_)), cmap = cm.Greys_r, interpolation ="nearest")
        plt.axis('off')     
    #plt.figure(2)
    plt.show()

    DIR2 = "../data/changedetection/camerajitter/traffic"
    srcDir = DIR2 + "/input"
    targetDir = DIR2 + "/grbm"
    scalevaraince = input("Enter value: ") #argmax(fmeasureV) + 12
    ii = 1 
    for k in os.listdir(srcDir):
        imf = Image.open(srcDir + '/' + k)
        print array(imf).shape
        data_test = (array(imf)).flatten()
        #dataset = np.vstack((dataset,d))
        
        data_test_0 = ((data_test)[::3]).T
        data_test_1 = ((data_test)[1::3]).T
        data_test_2 = ((data_test)[2::3]).T
        
        
        Nsteps = 5
        # evaluate 1st layer
        print 'evaluating 1...\t',
        
        # #datasub = zeros(x_*y_)
        #for point in xrange(Ndat):
            #X = asmatrix(data_test_0[:,point]).T
        X = asmatrix(data_test_0).T
        #dataout = vstack((dataout,X.flatten()))
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = dbn[0].forward(X) # self.activ(1)
            X = dbn[0].backward(Y,X)
        #print "S hsape:", X.shape
        #dataout = vstack((dataout,X.flatten()))
        dataout = subtract(asarray(X),data_test_0,asarray(dbn[0].vsigma),scalevaraince + 1 )
    
        #X = asmatrix(data_test_1[:,point]).T
        X = asmatrix(data_test_1).T
        #dataout1 = vstack((dataout1,X.flatten()))
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = dbn1[0].forward(X) # self.activ(1)
            X = dbn1[0].backward(Y,X)
        #print "S hsape:", X.shape
        #dataout1 = vstack((dataout1,X.flatten()))
        dataout1 = subtract(asarray(X),data_test_1,asarray(dbn1[0].vsigma),scalevaraince + 1)
    
        
        X = asmatrix(data_test_2).T
        #dataout2 = vstack((dataout2,X.flatten()))
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = dbn2[0].forward(X) # self.activ(1)
            X = dbn2[0].backward(Y,X)
        #print "S hsape:", X.shape
        #dataout2 = vstack((dataout2,X.flatten()))
        # plt.imshow(reshape(X,(x_,y_)))
        # plt.show()   
        dataout2 = subtract(asarray(X),data_test_2,asarray(dbn2[0].vsigma),scalevaraince + 1)
        
        # plt.imshow((reshape(data_test[::3,5],(x_,y_))), cmap = cm.Greys_r, interpolation ="nearest")
        # plt.axis('off')     
        # plt.show()

# plt.figure(1)
# for i in range(Ndat):
#     plt.subplot(5,5,i+1)
        d = multiply(asarray(dataout),asarray(dataout1),asarray(dataout2))
        d = mod(d+1,2)
        #print type(d[0])
        # #plt.savefig(targetDir + "\\" + str(ii) + ".png")
        #print 
        img_s = Image.fromarray(asarray(reshape(d,(x_,y_))*255,dtype="uint8"))
        # plt.figure(2)
        # plt.imshow(img_s)#,cmap = cm.Greys_r, interpolation ="nearest")
        # plt.show()   
        if (ii <1000):
            img_s.save(targetDir + "\\bin000" + str(ii) + ".bmp")
        else:
            img_s.save(targetDir + "\\bin00" + str(ii) + ".bmp")
        ii = ii +1 
        


if __name__ == '__main__':
  sys.exit(main(sys.argv))