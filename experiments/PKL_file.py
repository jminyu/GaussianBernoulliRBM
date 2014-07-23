'''
Created on 2014. 7. 10.

@author: Eric Schmits
'''

import sys
import os
import cPickle
import matplotlib.pylab as plt
import matplotlib.cm as cm
#import Image

 

import h5py as h5
import numpy as np
from numpy import *
from scipy import *
from numpy.core import *
from numpy.random import *
from scipy import stats, mgrid, c_, reshape, random, rot90
from scipy import ndimage
import pylab as p
from bsddb.test.test_dbtables import pickle

sys.path.append('../code')
sys.path.append('C:\Python27\Lib\site-packages')

from deepbelief import DBN, GaussianRBM, SemiRBM, RBM
from imageprocessing import Image




def make_pickle(x,dir_name,files):
	print dir_name
	dataset = np.zeros((320*240,3))
	for f in files:
		img = Image.open(dir_name+'/'+f)
		d =  (np.array(img.getdata())/255)
		dataset = np.vstack((dataset,d))
	output = open(dir_name+'.pkl','wb')
	cPickle.dump(dataset, output)
	output.close()	



def print_it(x, dir_name, files):
    print dir_name
    dataset = np.zeros(120*160)
    print 'image pkl processing.....-print_it'
    #dataset = np.zeros(256*320)
    for f in files:
        imf = Image.open(dir_name + '/' + f)
        d = ((np.array(imf.getdata())/255) > 0.5 ) * 1
        dataset = np.vstack((dataset,d))
    output = open(dir_name + '.pkl', 'wb')
    cPickle.dump(dataset, output)
    print 'processing fin!-print_it'
    output.close()

def generate_orignal(x, dir_name, files):
    print dir_name
    dataset = np.zeros(120*160)
    print 'image orignal processing.....-generate_orignal'
    for f in files:
        imf = Image.open(dir_name + '/' + f)
        d = (np.array(imf.getdata())/255)  
        dataset = np.vstack((dataset,d))
    print 'processing fin!'
    output = open(dir_name + '_orignal.pkl', 'wb')
    cPickle.dump(dataset[1:, :], output)
    output.close()
        
if __name__ == '__main__':
	DIR2 = "../data/DAcam/sample1"
	changeDetectionDir = "../data/changedetection/camerajitter/traffic/inputPKL"
  	#DIR2 = "../data/DAcam/sample1/Color"
  	depthDir = DIR2 + "/depthPKL"
  	colorDir = DIR2 + "/colorPKL"
  	#targetDir = DIR2 + "/grbm"
  	'''
    #for depth processing
    
  	os.path.walk(colorDir,print_it,0)
  	os.path.walk(depthDir, generate_orignal, 0)
  	'''
	#os.path.walk(depthDir,print_it,0)
  	os.path.walk(changeDetectionDir,make_pickle,0)
  	print 'procedure is finish!'
  	sys.exit()	
