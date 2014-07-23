import sys
import cPickle
import matplotlib.pylab as plt
import matplotlib.cm as cm
import Image
import os


from numpy import *
from numpy.random import *
import pylab as p
from scipy import stats, mgrid, c_, reshape, random, rot90
#x_ = 350
#y_ = 640

DIR = "C:\work\\backgdSubt\dataset\datasets\change detection\\camerajitter\\sidewalk"
srcDir = DIR + "\dpgmm"
targetDir = DIR + "\dpgmmalt"

img = Image.open(DIR + "\ROI.bmp")
print "Doie : " , (asarray(img)[180,180:200])
groundtruth = ((asarray(img)[:,:]))#.flatten()
p.imshow(img)
p.show()
print 
for k in os.listdir(srcDir):
    imf = Image.open(srcDir + '/' + k)
    print array(imf).shape
    data_test = (asarray(imf)[:,:,0]/255)#.flatten()
    #dataset = np.vstack((dataset,d))
    #p.imshow(reshape(data_test,(x_,y_)))
    #p.show()
        
    d = multiply(asarray(data_test),asarray(groundtruth))
    
    img_s = Image.fromarray(d*255)#asarray(reshape(d,(x_,y_))*255,dtype="uint8"))
    # plt.figure(2)
    # plt.imshow(img_s)#,cmap = cm.Greys_r, interpolation ="nearest")
    # plt.show()   
    #if (ii <1000):
    img_s.save(targetDir + "//" + k)
    #else:
    #    img_s.save(targetDir + "\\bin00" + str(ii) + ".bmp")
    #ii = ii +1 