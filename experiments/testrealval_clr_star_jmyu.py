import sys
import cPickle
import matplotlib.pylab as plt
import matplotlib.cm as cm
import Image
import h5py as h5

sys.path.append('../code')

from deepbelief import DBN, GaussianRBM, SemiRBM, RBM
from numpy import *
from numpy.random import *
import pylab as p
from scipy import stats, mgrid, c_, reshape, random, rot90
from scipy import ndimage
import datetime




x_ = 120
y_ = 160
#runtime = datetime.datetime.now().strftime('%H:%M:%S %d,%b,%Y')
#C:\Python27\Lib\site-packages



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
    noOfExamples = train_set.shape[0]-23
    train_set_x = train_set[:noOfExamples,:]
    #train_set_x = train_set[243:257,:] 
    test_set_x = train_set[noOfExamples:,:]
    train_set_x0 = train_set_x[:,::3]
    train_set_x1 = train_set_x[:,1::3]
    train_set_x2 = train_set_x[:,2::3]
    #print test_set_x,train_set_x
    return train_set_x0.T,train_set_x1.T,train_set_x2.T, test_set_x.T

def subtract(x,y,sigma,scalesigma):
    #scalesigma = scalesigma + 10
    x = x.flatten()
    y = y.flatten()
    sigma = sigma.flatten()
    a = (x+sigma*scalesigma)>y
    b = (x-sigma*scalesigma)<y
    return a*b*1
    
def storeWeights(id, filename, W, sigma, W1, sigma1, W2, sigma2):
    f = h5.File(filename + ".hdf5")
    group = f.create_group(id)
    group.create_dataset('sigma',data=sigma)
    group.create_dataset("weights",data = W)
    group.create_dataset('sigma1',data=sigma1)
    group.create_dataset("weights1",data = W1)
    group.create_dataset('sigma2',data=sigma2)
    group.create_dataset("weights2",data = W2)
    f.close()

def f_measure(fg,gt):
    fg_pixgt = where(gt == 1)
    bg_pixgt = where(gt == 0)

    fg_pixfg = where(fg == 1)
    bg_pixfg = where(fg == 0)


    tp = (where(gt[fg_pixgt] == fg[fg_pixgt]))[0]
    fp = (where(gt[fg_pixgt] != fg[fg_pixgt]))[0]
    tn = (where(gt[bg_pixgt] == fg[bg_pixgt]))[0]
    fn = (where(gt[bg_pixgt] != fg[bg_pixgt]))[0]

    DR = 0
    percision = 0
    if ((len(tp) + len(fn)) > 0):
        DR = len(tp) * 1.0 / ( len(tp) + len(fn) )
    if ((len(tp) + len(fp)) > 0):
        percision = len(tp) * 1.0 / (len(tp) + len(fp)) 
    if ((DR + percision) > 0):
        F = 2 * DR * percision / (DR + percision)
    else:
        F = 0
    return F

def main(argv):
    # load preprocessed data samples
    print 'loading data...\t',
    #data_train, data_test = genData() #load('../data/vanhateren.npz')
    category = "office"
    data_train_0,data_train_1,data_train_2, data_test = load_data_BS("..\data\watersurface\orignal_color.pkl")
    img = Image.open(".\data\watersurface\hand_segmented_01850.bmp")
    x_, y_ = (asarray(img)[:,:,0]).shape
    print "Doie : " , (asarray(img)[:,:,0]).shape
    groundtruth = (((asarray(img)[:,:,0])/255.0 > 0.5) * 1).flatten()
    
    #groundtruth = (((asarray(img)[:,:])/255.0 > 0.5) * 1).flatten()
    print '[DONE]'  
    

    print data_test.shape
    # remove DC component (first component)
    # data_train = data['train'][1:, :]
    # data_test = data['test'][1:, :]

    # create 1st layer
    dbn = DBN(GaussianRBM(num_visibles=data_train_0.shape[0], num_hiddens=8))
    dbn1 = DBN(GaussianRBM(num_visibles=data_train_1.shape[0], num_hiddens=8))
    dbn2 = DBN(GaussianRBM(num_visibles=data_train_2.shape[0], num_hiddens=8))
    
    dbn[0].learning_rate = 0.001
    dbn1[0].learning_rate = 0.001
    dbn2[0].learning_rate = 0.001

   

    # train 1st layer
    print 'training...\t',
    dbn.train(data_train_0, num_epochs=5, batch_size=1,shuffle=False)
    dbn1.train(data_train_1, num_epochs=5, batch_size=1,shuffle=False)
    dbn2.train(data_train_2, num_epochs=5, batch_size=1,shuffle=False)
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
       
        #open_img = ndimage.binary_opening(reshape(d,(x_,y_)))
        #close_img = ndimage.binary_closing(open_img)
        m_filter = ndimage.median_filter(reshape(d,(x_,y_)),8)
        # plt.figure(3)
        # plt.imshow(close_img)
        # plt.show()
        d = m_filter.flatten()
        print "Image Example Fmeaure: ",i," : ", f_measure(d,groundtruth) * 100
        # d[0::3] = asarray(dataout[i+1,:])
        # d[1::3] = asarray(dataout1[i+1,:])
        # d[2::3] = asarray(dataout2[i+1,:])
        # d[:,:,0] = (reshape(asarray(dataout[i+1,:]),(x_,y_)))
        # d[:,:,1] = (reshape(asarray(dataout1[i+1,:]),(x_,y_)))
        # d[:,:,2] = (reshape(asarray(dataout2[i+1,:]),(x_,y_)))
        #img_s = Image.fromarray(asarray(reshape(d,(x_,y_))*255,dtype="uint8"))
        #img_s.save("C:\work\\backgdSubt\dataset\datasets\change detection\\baseline\\baseline\office\grbm" + "\\" + str(i) + ".bmp")
        plt.imshow(reshape(d,(x_,y_)), cmap = cm.Greys_r, interpolation ="nearest")
        plt.axis('off')     
    plt.figure(2)
    

    for k in range(8):
        plt.subplot(4,2,k+1)
        d = zeros((x_*y_*3))
        d[0::3] = asarray(dbn[0].W[:,k].flatten())
        d[1::3] = asarray(dbn1[0].W[:,k].flatten())
        d[2::3] = asarray(dbn2[0].W[:,k].flatten())
        plt.imshow(reshape(d,(x_,y_,3)))#, cmap = cm.Greys_r, interpolation ="nearest")
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


if __name__ == '__main__':
  sys.exit(main(sys.argv))
