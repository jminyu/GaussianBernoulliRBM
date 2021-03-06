import sys
import cPickle
import matplotlib.pylab as plt
import matplotlib.pyplot
import matplotlib.cm as cm


sys.path.append('../code')


from deepbelief import GaussianRBM_ACL

from numpy import *
from numpy.random import *
import pylab as p
from scipy import stats, mgrid, c_, reshape, random, rot90

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
    f.close()
    print train_set.shape
    train_set_x = train_set[:240,:]
    #train_set_x = train_set[243:257,:] 
    test_set_x = train_set[242:258,:]         
    return train_set_x, test_set_x

def main(argv):
    # load preprocessed data samples
    print 'loading data...\t',
    #data_train, data_test = genData() #load('../data/vanhateren.npz')
    data_train, data_test = load_data_BS("..\data\camouflage\orignal_color.pkl")
    print '[DONE]'
    


    # remove DC component (first component)
    # data_train = data['train'][1:, :]
    # data_test = data['test'][1:, :]

    # create 1st layer
    grbm = GaussianRBM_ACL(num_visibles=data_train.shape[1], num_hiddens=8)

    # hyperparameters
    grbm.learning_rate = 0.001
    #dbn[0].weight_decay = 1E-2
    grbm.cd_steps = 15
    # dbn[0].persistent = True

    # train 1st layer
    print 'training...\t',
    grbm.train(data_train, epocs=4)
    print '[DONE]'

    Ndat = 1000
    Nsteps = 1
    # evaluate 1st layer
    print 'evaluating...\t',
    datout = zeros( (2,Ndat), dtype=float32)
    for point in xrange(Ndat):
        X = data_test[:,point]
        #print "testing:", X.shape
        for recstep in xrange(Nsteps): 
            Y = grbm.forward(X) # self.activ(1)
            X = grbm.backward(Y,X)
        #print "S hsape:", X.shape
        plt.imshow((reshape(data_test[:,point],(120,160))), cmap = cm.Greys_r, interpolation ="nearest")
        plt.axis('off')     
        plt.figure(2)
        for i in range(8):
            plt.subplot(4,2,i+1)
            plt.imshow((reshape(X,(120,160))), cmap = cm.Greys_r, interpolation ="nearest")
            plt.axis('off')     
        # plt.figure(3)
        # plt.imshow((reshape(dbn[0].vsigma,(120,160))), cmap = cm.Greys_r, interpolation ="nearest")
        plt.show()    

    # p.figure(1)
    # p.plot(data_train[0,:],data_train[1,:], 'b.')
    # p.axis([0.0, 1.0, 0.0, 1.0])

    # p.figure(2)
    # p.plot(data_test[0,:],data_test[1,:], 'b.')
    # p.axis([0.0, 1.0, 0.0, 1.0])
    
    # p.figure(3)
    # p.plot(datout[0,:],datout[1,:], 'b.')
    # p.axis([0.0, 1.0, 0.0, 1.0])

    # p.figure(4)
    # p.plot(datout[0,:],datout[1,:], 'b.')
    
    # #p.axis([0.0, 1.0, 0.0, 1.0])

    # #p.figure(4)
    # #p.hist(datout)
    # print dbn[0].vsigma
    # p.show()    

    # logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
    # loglik = dbn.estimate_log_likelihood(data_test)
    # print '[DONE]'
    # print
    # print 'estimated log-partf.:\t', logptf
    # print 'estimated log-loss:\t', -loglik / data_test.shape[0] / log(2)
    # print



#   # create 2nd layer
#   dbn.add_layer(SemiRBM(num_visibles=100, num_hiddens=100))

#   # initialize parameters
#   dbn[1].L = dbn[0].W.T * dbn[0].W
#   dbn[1].b = dbn[0].W.T * dbn[0].b + dbn[0].c + 0.5 * asmatrix(diag(dbn[1].L)).T
#   dbn[1].L = dbn[1].L - asmatrix(diag(diag(dbn[1].L)))

#   # hyperparameters
#   dbn[1].learning_rate = 5E-3
#   dbn[1].learning_rate_lateral = 5E-4
#   dbn[1].weight_decay = 5E-3
#   dbn[1].weight_decay_lateral = 5E-3
#   dbn[1].momentum = 0.9
#   dbn[1].momentum_lateral = 0.9
#   dbn[1].num_lateral_updates = 20
#   dbn[1].damping = 0.2
#   dbn[1].cd_steps = 1
#   dbn[1].persistent = True

#   # train 2nd layer
#   print 'training...\t',
#   dbn.train(data_train, num_epochs=100, batch_size=100)
#   print '[DONE]'

#   # evaluate 2nd layer
#   print 'evaluating...\t',
#   logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
#   loglik = dbn.estimate_log_likelihood(data_test, num_samples=100)
#   print '[DONE]'
#   print
#   print 'estimated log-partf.:\t', logptf
#   print 'estimated log-loss:\t', -loglik / data_test.shape[0] / log(2)
#   print



#   # fine-tune with wake-sleep
#   dbn[0].learning_rate /= 4.
#   dbn[1].learning_rate /= 4.

#   print 'fine-tuning...\t',
#   dbn.train_wake_sleep(data_train, num_epochs=10, batch_size=10)
#   print '[DONE]'

#   # reevaluate
#   print 'evaluating...\t',
#   logptf = dbn.estimate_log_partition_function(num_ais_samples=100, beta_weights=arange(0, 1, 1E-3))
#   loglik = dbn.estimate_log_likelihood(data_test, num_samples=100)
#   print '[DONE]'
#   print
#   print 'estimated log-partf.:\t', logptf
#   print 'estimated log-loss:\t', -loglik / data_test.shape[0] / log(2)

#   return 0



if __name__ == '__main__':
  sys.exit(main(sys.argv))
