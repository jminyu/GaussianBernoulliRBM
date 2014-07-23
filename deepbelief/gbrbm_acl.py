import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

class GaussianRBM_ACL():
    """
    An implementation of the Gaussian RBM with continuous visible nodes.

    """

    def __init__(self, num_visibles, num_hiddens):
        #AbstractBM.__init__(self, num_visibles, num_hiddens)

        # hyperparameters
        self.sigma = 0.2
        
        self.learning_rate = 0.01
        self.weight_decay = 0.001
        self.momentum = 0.5

        self.cd_steps = 10
        self.persistent = False

        self.sparseness = 0.0
        self.sparseness_target = 0.1

        #self.sampling_method = AbstractBM.GIBBS

        # relevant for HMC sampling
        self.lf_steps = 10
        self.lf_step_size = 0.01
        self.lf_adaptive = True

        # parameters
        self.W = np.random.randn(num_visibles, num_hiddens) / (num_visibles + num_hiddens)
        self.b = np.zeros(num_visibles)
        self.c = np.zeros(num_hiddens) - 1.
        self.vsigma = np.ones(num_visibles)
                
        # increments
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dc = np.zeros_like(self.c)
        self.dvsigma = np.ones_like(self.vsigma)

        # variables
        self.X = np.zeros(num_visibles)
        self.Y = np.zeros(num_hiddens)

        # probabilities
        self.P = np.zeros_like(self.X)
        self.Q = np.zeros_like(self.Y)

        # states of persistent Markov chain
        self.pX = np.zeros([num_visibles, 100])
        self.pY = np.zeros([num_hiddens, 100])

        # used by annealed importance sampling
        self.ais_logz = None
        self.ais_samples = None
        self.ais_logweights = None



    def forward(self, X=None):
        #self.Q = 1. / (1. + np.exp(-self.W.T * X ))#/ self.sigma - self.c))
        self.Q = 1. / (1. + np.exp(- np.dot(X/np.square(self.vsigma),self.W) - self.c))
        self.Y = (np.random.rand(*self.Q.shape) < self.Q).astype(self.Q.dtype)
        return self.Q.copy(), self.Y.copy()



    def backward(self, Y=None, X=None):
        #self.X = (self.sigma * self.W * Y + self.b + self.sigma * np.random.randn(self.X.shape[0], Y.shape[1]))
        self.X = (np.multiply(self.vsigma,np.dot(self.W, Y) + self.b )) #+ np.multiply(self.vsigma,np.random.randn(self.X.shape[0], Y.shape[1])))
        return self.X.copy()




    def _free_energy_gradient(self, X):
        Q = 1. / (1. + np.exp(-np.dot(X/self.sigma ,self.W) - self.c))
        return (X - self.b) / (self.sigma * self.sigma) - self.W * Q



    # def _centropy_hid_vis(self, X):
    #     # compute conditional probabilities of hidden units being active
    #     Q = 1. / (1. + np.exp(-self.W.T * np.asmatrix(X) / self.sigma - self.c))

    #     A = np.multiply(Q, np.log(Q))
    #     B = np.multiply(1. - Q, np.log(1. - Q))

    #     # zero times infinity gives zero
    #     A[Q == 0] = 0
    #     B[Q == 1] = 0

    #     # integrate
    #     return -np.sum(A + B, 0)

    def train(self, train_data, epocs=10):
        """
        Trains the parameters of the BM on a batch of data samples. The
        data stored in C{X} is used to estimate the likelihood gradient and
        one step of gradient ascend is performed.

        @type  X: array_like
        @param X: example states of the visible units
        """
        #print "I am traing sigma"
        tr_ex = train_data.shape[0]
        
        for epoc in range(epocs):
            for ex in range(tr_ex):    
                X = train_data[ex]
                Xtemp = X.copy()
                # positive phase
                Yprob, Y = self.forward(X)
                Ytemp = Y.copy()
                # store posterior probabilities
                Q = self.Q.copy()
                
                pos = np.outer(X,Y)
                posSigma = (np.square(Xtemp - self.b) - 2.0*Xtemp*np.dot(Ytemp,self.W.T)).sum(axis=0)/(self.vsigma**3)
                #grad_sigma = (((v - visibleBias)**2 - 2.0 * v * dot(h, weights.T)).sum(axis=0) / (sigma**3))
                # if self.persistent:
                #     self.X = self.pX
                #     self.Y = self.pY

                # negative phase
                for t in range(self.cd_steps):
                    #print 'cd', t
                    X = self.backward(Y)
                    Y,Yprob = self.forward(X)

                # if self.persistent:
                #     self.pX = self.X.copy()
                #     self.pY = self.Y.copy()
                # print max(X),max(self.b)
                
                negSigma = (np.square(X - self.b) - 2.0*X*np.dot(Y,self.W.T)).sum(axis=0)/(self.vsigma**3)
                # update parameters
                
                self.dW = pos / X.shape[0] - np.outer(X,Y) / self.X.shape[0] \
                        - self.weight_decay * self.W \
                        + self.momentum * self.dW
                self.dvsigma = posSigma/X.shape[0] - negSigma/X.shape[0] #- self.weight_decay*self.vsigma + self.momentum*self.dvsigma
                #print "sigma Gradient: ", self.dvsigma
                self.db = Xtemp.mean(0) - self.X.mean(0) + self.momentum * self.db
                self.dc = Q.mean(0) - self.Q.mean(0) + self.momentum * self.dc
        #               - self.sparseness * np.multiply(np.multiply(Q, 1. - Q).mean(1), (Q.mean(1) - self.sparseness_target))

                self.W += self.dW * self.learning_rate
                self.vsigma += self.dvsigma * 0.000001
                self.b += self.db * self.learning_rate
                self.c += self.dc * self.learning_rate

            weight = self.W
            print weight.shape
            for i in range(8):
              plt.subplot(4,2,i+1)
              plt.imshow((np.reshape(weight[:,i],(120,160))), cmap = cm.Greys_r, interpolation ="nearest")
              plt.axis('off')     
            plt.show()