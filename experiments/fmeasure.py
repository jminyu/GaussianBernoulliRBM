from numpy import *
import os.path
import Image
import matplotlib.pylab as plt


def f_measure(f, fg,gt):
    fg_pixgt = where(gt == 1)
    bg_pixgt = where(gt == 0)

    fg_pixfg = where(fg == 1)
    bg_pixfg = where(fg == 0)


    tp = (where(gt[fg_pixgt] == fg[fg_pixgt]))[0]
    fp = (where(gt[fg_pixgt] != fg[fg_pixgt]))[0]
    tn = (where(gt[bg_pixgt] == fg[bg_pixgt]))[0]
    fn = (where(gt[bg_pixgt] != fg[bg_pixgt]))[0]


    DR = len(tp) * 1.0 / ( len(tp) + len(fn) )
    percision = len(tp) * 1.0 / (len(tp) + len(fp)) 
    F = 2 * DR * percision / (DR + percision)
    print f, " fmeasure:", F, " tp:fp:tn:fn =", len(tp),":",len(fp),":",len(tn),":",len(fn), "Recall:", DR,"  Percision:", percision
    return F

def print_it(x, dir_name, files):
    global dataset
    print "here"
    #dataset = []
    #dataset = np.zeros(256*320)
    for f in files:
        imf = Image.open(dir_name + '/' + f)
        #d = np.array(imf)
        d = ((array(imf.getdata())/255) > 0.5 ) * 1 
        #print f
        #print d.shape
        dataset.append(d)
    return dataset

def print_fmeasure(x, dir_name, files):
    global dataset
    print "here"
    i = 0
    s = 0 
    for f in files:
        imf = Image.open(dir_name + '/' + f)
        #d = np.array(imf)
        
        groundtruth1 = (((asarray(imf)[:,:,0])/255.0 > 0.5) * 1).flatten()
        #print "shape:", groundtruth1.shape, "shape2:", dataset[i].shape
        #print i
        #   print f
        s = s + f_measure(f, groundtruth1, dataset[i])
        i = i + 1
    print "avg: ", s/i 


# img = Image.open("data/changedetection\\baseline\highway/gt001367.png")
# img1 = Image.open("data/changedetection\\baseline\highway/bin001367.png")


# print (asarray(img1)).shape
# plt.imshow(((asarray(img)[:,:])/255.0 > 0.5)* 1)
# plt.figure(2)
# plt.imshow(((asarray(img1)[:,:])/255.0 > 0.5) * 1)

# plt.show()

# print "Doie : " , (asarray(img)[:,:]).shape
# groundtruth = (((asarray(img)[:,:])/255.0 > 0.5) * 1).flatten()
# groundtruth1 = (((asarray(img1)[:,:,0])/255.0 > 0.5) * 1).flatten()

# f_measure(groundtruth1,groundtruth)
dataset = []
os.path.walk('C:\work\\backgdSubt\GRBM\deepbelief\code\data\cdresults\GT', print_it, 0)
os.path.walk('C:\work\\backgdSubt\GRBM\deepbelief\code\data\cdresults\DPGMM', print_fmeasure, 0)    