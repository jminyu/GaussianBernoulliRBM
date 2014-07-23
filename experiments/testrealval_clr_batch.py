import sys
import cPickle
import matplotlib.pylab as plt
import matplotlib.cm as cm
import Image
import os

sys.path.append('../code')

from deepbelief import DBN, GaussianRBM, SemiRBM, RBM
from numpy import *
from numpy.random import *
import pylab as p
from scipy import stats, mgrid, c_, reshape, random, rot90

x_ = 480
y_ = 704
batch_no = 0
batch_size = 26

#train_set_x0,train_set_x1,train_set_x2

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
    f = open(dataset, 'rb')    
    train_set= cPickle.load(f)
    f.close()
    print train_set.shape
    train_set_x = train_set[:198,:]
    #train_set_x = train_set[243:257,:] 
    test_set_x = train_set[198:,:]
    train_set_x0 = train_set_x[:,::3]
    train_set_x1 = train_set_x[:,1::3]
    train_set_x2 = train_set_x[:,2::3]
    #print test_set_x,train_set_x
    return train_set_x0.T,train_set_x1.T,train_set_x2.T, test_set_x.T

def subtract(x,y,sigma,scalesigma):
    #scalesigma = scalesigma +10
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

    F = 0 
    tp = (where(gt[fg_pixgt] == fg[fg_pixgt]))[0]
    fp = (where(gt[fg_pixgt] != fg[fg_pixgt]))[0]
    tn = (where(gt[bg_pixgt] == fg[bg_pixgt]))[0]
    fn = (where(gt[bg_pixgt] != fg[bg_pixgt]))[0]

    if (( len(tp) + len(fn) ) > 0 and (len(tp) + len(fp)) > 0):
        DR = len(tp) * 1.0 / ( len(tp) + len(fn) )
        percision = len(tp) * 1.0 / (len(tp) + len(fp))

        F = 2 * DR * percision / (DR + percision)
    return F

def load_batch(x, dir_name, files):
    #print dir_name
    global batch_no
    global batch_size
    dataset = zeros(x_*y_*3, dtype=uint8)
    #dataset = np.zeros(120*160)
    start = batch_no * batch_size
    for f in range(batch_size):
        #print dir_name + '/' + files[start+f]
        imf = Image.open(dir_name + '/' + files[start+f])
        #print array(imf).shape
        d = (array(imf)).flatten()
        dataset = vstack((dataset,d))
    
    global train_set_x0
    global train_set_x1
    global train_set_x2
    
    train_set_x0 = dataset[1:,::3]
    train_set_x1 = dataset[1:,1::3]
    train_set_x2 = dataset[1:,2::3]
    
    #return train_set_x0,train_set_x1,train_set_x2
    # output = open(dir_name + '_color.pkl', 'wb')
    # cPickle.dump(dataset[1:,:], output)
    # output.close()

def load_test_batch(x, dir_name, files):
    #print dir_name
    dataset = zeros(x_*y_*3, dtype=uint8)
    #dataset = np.zeros(120*160)
    
    for f in files:
        #print dir_name + '/' + files[start+f]
        imf = Image.open(dir_name + '/' + f)
        #print array(imf).shape
        d = (array(imf)).flatten()
        dataset = vstack((dataset,d))
    
    global data_test
    data_test = dataset.T

    

def main(argv):
    # load preprocessed data samples
    print 'loading data...\t',
    #data_train, data_test = genData() #load('../data/vanhateren.npz')
    # data_train_0,data_train_1,data_train_2, data_test = load_data_BS("data/changedetection\intermitentmotion\streetlight/orignal_color.pkl")
    img = Image.open("data/changedetection\ptz\continuous/gt000962.png")
    print "Doie : " , (asarray(img)[:,:]).shape
    groundtruth = (((asarray(img)[:,:])/255.0 > 0.5) * 1).flatten()
    
    # #groundtruth = (((asarray(img)[:,:])/255.0 > 0.5) * 1).flatten()
    # print '[DONE]'
    batches = 3


    # remove DC component (first component)
    # data_train = data['train'][1:, :]
    # data_test = data['test'][1:, :]

    # create 1st layer
    dbn = DBN(GaussianRBM(num_visibles=x_*y_, num_hiddens=15))
    dbn1 = DBN(GaussianRBM(num_visibles=x_*y_, num_hiddens=15))
    dbn2 = DBN(GaussianRBM(num_visibles=x_*y_, num_hiddens=15))
    
    dbn[0].learning_rate = 0.0001
    dbn1[0].learning_rate = 0.0001
    dbn2[0].learning_rate = 0.0001


    # train 1st layer
    print 'training...\t',
    for k in range(2):
        print "epoc :", k
        for i in range(batches):
            global batch_no
            batch_no = i
            os.path.walk('C:\work\\backgdSubt\dataset\changedetection\ptz\continuous\\orignal', load_batch, 0)
            global train_set_x0
            global train_set_x1
            global train_set_x2

            dbn.train(train_set_x0.T, num_epochs=1, batch_size=1,shuffle=False)
            dbn1.train(train_set_x1.T, num_epochs=1, batch_size=1,shuffle=False)
            dbn2.train(train_set_x2.T, num_epochs=1, batch_size=1,shuffle=False)
    
    print '[DONE]'

    os.path.walk('C:\work\\backgdSubt\dataset\changedetection\ptz\continuous\\test', load_test_batch, 0)

    data_test_0 = ((data_test.T)[:,::3]).T
    data_test_1 = ((data_test.T)[:,1::3]).T
    data_test_2 = ((data_test.T)[:,2::3]).T
    
    Ndat = 25 #data_test_0.shape[1]
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

    plt.figure(1)
    for i in range(Ndat):
        plt.subplot(5,5,i+1)
        d = multiply(asarray(dataout[i+1,:]),asarray(dataout1[i+1,:]),asarray(dataout2[i+1,:]))
        d = mod(d+1,2)
        #print "Image Example Fmeaure: ",i," : ", f_measure(d,groundtruth) * 100
        # d[0::3] = asarray(dataout[i+1,:])
        # d[1::3] = asarray(dataout1[i+1,:])
        # d[2::3] = asarray(dataout2[i+1,:])
        # d[:,:,0] = (reshape(asarray(dataout[i+1,:]),(x_,y_)))
        # d[:,:,1] = (reshape(asarray(dataout1[i+1,:]),(x_,y_)))
        # d[:,:,2] = (reshape(asarray(dataout2[i+1,:]),(x_,y_)))
        plt.imshow(reshape(d,(x_,y_)), cmap = cm.Greys_r, interpolation ="nearest")
        plt.axis('off')     
    plt.figure(2)
    

    for k in range(15):
        plt.subplot(5,3,k+1)
        d = zeros((x_*y_*3))
        d[0::3] = asarray(dbn[0].W[:,k].flatten())
        d[1::3] = asarray(dbn1[0].W[:,k].flatten())
        d[2::3] = asarray(dbn2[0].W[:,k].flatten())
        plt.imshow(reshape(d[0::3],(x_,y_)), cmap = cm.Greys_r, interpolation ="nearest")
        plt.axis('off')     
    # plt.figure()
    # plt.imshow((reshape(dbn[0].vsigma[:19200],(x_,y_))))
    
    # plt.figure(2)
    # plt.imshow((reshape(dbn[0].vsigma[19200:19200*2],(x_,y_))))
    
    # plt.figure(3)
    # plt.imshow((reshape(dbn[0].vsigma[19200*2:19200*3],(x_,y_))))
    
    plt.figure(3)
    print type(dbn[0].vsigma)
    plt.imshow(reshape(asarray(dbn[0].vsigma),(x_,y_)))
    plt.show()    

    
    print dbn[0].vsigma
    p.show()    

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
