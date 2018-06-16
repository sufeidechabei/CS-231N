from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = np.random.normal(0,weight_scale,size=(input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0,weight_scale,size=(hidden_dim,num_classes))
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        scores = None
        hidden,affine_relu_cache = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        out,fc_cache = affine_forward(hidden,self.params['W2'],self.params['b2'])
        scores = out
        if y is None:
            return scores
        loss,grads = 0,{}
        loss,dout = softmax_loss(out,y)
        loss += 0.5*self.reg*np.sum(self.params['W1']*self.params['W1']) + 0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
        dhidden,dw2,db2 = affine_backward(dout,fc_cache)
        grads['W2'] = dw2 + self.reg*self.params['W2']
        grads['b2'] = db2
        dx,dw1,db1 = affine_relu_backward(dhidden,affine_relu_cache)
        grads['W1'] = dw1 + self.reg*self.params['W1']
        grads['b1'] = db1
        return loss,grads
        
        
        

   


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.hidden_dims = hidden_dims
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.num_layers = 1 + len(hidden_dims)
        for x in range(len(hidden_dims)):
            if x==0:
                self.params['W'+str(x+1)] = np.random.normal(0,weight_scale,size=(input_dim,hidden_dims[x]))
                self.params['b'+str(x+1)] = np.zeros(hidden_dims[x])
            else:
                self.params['W'+str(x+1)] = np.random.normal(0,weight_scale,size=(hidden_dims[x-1],hidden_dims[x]))
                self.params['b'+str(x+1)] = np.zeros(hidden_dims[x])
        self.params['W'+ str(len(hidden_dims)+1)] = np.random.normal(0,weight_scale,size=(hidden_dims[x],num_classes))
        self.params['b'+ str(len(hidden_dims)+1)] = np.zeros(num_classes)
        self.gammabetacache = []
        
        
                
            

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
                

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            for i in range(self.num_layers-1):
                D = self.hidden_dims[i]
                gamma = np.random.randn(D)
                beta = np.random.randn(D)
                self.params['gamma' + str(i+1)] = gamma
                self.params['beta' + str(i+1)] = beta
          
                
                
                
            

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
                self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
   
        

        scores = None
        X = X.reshape(X.shape[0],-1)
        hidden = X
        cache = {}
        drop_cache = {}
        batchcaches = []
        if mode =='train':
            for i in range(self.num_layers - 1):
                if self.use_batchnorm ==True:
                    gamma = self.params['gamma' + str(i+1)]
                    beta = self.params['beta' + str(i+1)]
                    print(hidden.shape)
                    _,cache[i] = affine_relu_forward(hidden,self.params['W' + str(i + 1)],self.params['b' + str(i+1)])
                    hidden,batchcache = affine_bn_relu_forward(hidden,self.params['W'+ str(i + 1)],self.params['b' + str(i+1)],gamma,beta,self.bn_params[i])
          
                    batchcaches.append(batchcache)
                else:
                    hidden,cache[i] = affine_relu_forward(hidden,self.params['W' + str(i + 1)],self.params['b' + str(i+1)])
                if self.use_dropout == True :
                     hidden,drop_cache[i] = dropout_forward(hidden,self.dropout_param)
          
                     
                     
                        
                        
            out,cache[self.num_layers - 1] = affine_forward(hidden,self.params['W'+str(self.num_layers)],self.params['b'+ str(self.num_layers)] )
            scores = out
        if mode =='test':
            for i in range(self.num_layers-1):
                if self.use_batchnorm == True:
                    gamma = self.params['gamma' + str(i+1)]
                    beta = self.params['beta'+ str(i+1)]
                    hidden,bn_param = affine_bn_relu_forward(hidden,self.params['W' + str(i+1)],self.params['b' + str(i+1)],gamma,beta,self.bn_params[i])
                else:                                           
                    hidden,cache[i] = affine_relu_forward(hidden,self.params['W'+ str(i+1)],self.params['b' + str(i+1)])
                
                   
                    
            
            out,cache[self.num_layers - 1] = affine_forward(hidden,self.params['W' + str(self.num_layers)],self.params['b' + str(self.num_layers)])
            return out
            
            
            
        
        
                
            
            

                
                
                
                
            
            
            
            
             
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early

        loss, grads = 0.0, {}
        loss,dout = softmax_loss(out,y)    
        dhidden,dw,db = affine_backward(dout,cache[self.num_layers-1])
        loss += 0.5 *self.reg *np.sum(self.params['W'+str(self.num_layers)]*self.params['W'+str(self.num_layers)])    
        grads['W'+str(self.num_layers)] = dw + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db
        for i in range(self.num_layers - 1,0,-1):
            loss += 0.5*self.reg*np.sum(self.params['W'+str(i)]*self.params['W'+str(i)]) 
            if self.use_dropout == True and i>1:
                dhidden = dropout_backward(dhidden,drop_cache[i-1])
            if self.use_batchnorm == True:
                dz,grads['gamma' + str(i) ],grads['beta' + str(i)] = affine_bn_relu_backward(dhidden,batchcaches[i-1],cache[i-1][1])
                dhidden,dw,db = affine_backward(dz,cache[i-1][0])
            else:
                dhidden,dw,db = affine_relu_backward(dhidden,cache[i-1])
            
            grads['W'+str(i)] = dw + self.reg*self.params['W'+ str(i)]
            grads['b'+str(i)] = db
        return loss,grads
    
        
        
 
            
               
                
                
                
                
                
                
                
                
               
                
            
            
        
        
        
         
        
        
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

