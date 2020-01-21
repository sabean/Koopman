""" EDMD with dictionary learning """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import autograd.numpy as np
import time
import scipy.optimize
from autograd import grad 

class EDMD_DL:
    def __init__(self):
        self.__build_flag = -1
        self.__train_flag = -1

    def build(self, dim, ddim, hdim, num_layers=3, 
              activation=tf.nn.tanh, init_lr=1e-5,
              random_state=115):
        """ Build tensorflow model """

        # Parameters
        self.__random_state = random_state
        self.__dim = dim
        self.__ddim = ddim
        self.__hdim = hdim
        self.__num_layers = num_layers

        kmatdim = ddim + 1 + dim
        num_layers = self.__num_layers
        std = 1.0 / np.sqrt(hdim)
        std_proj = 1.0

        # Form B matrix
        self.__B = np.zeros((dim+1+ddim, dim))
        for i in range(dim):
            self.__B[i+1][i] = 1

        tf.reset_default_graph()
        self.__tf_nlr = activation
        # Constants
        # TF_PI = tf.constant(value=np.pi, dtype=tf.float64)
        tf.set_random_seed(self.__random_state)
        # Build graph

        # Input data placeholders
        self.__x1 = tf.placeholder(tf.float64, [None, dim])
        self.__y1 = tf.placeholder(tf.float64, [None, dim])
        self.__x2 = tf.placeholder(tf.float64, [None, dim])
        self.__y2 = tf.placeholder(tf.float64, [None, dim])

        # K matrices
        with tf.variable_scope("K1_matrix"):
            self.__K1 = tf.get_variable(shape=(kmatdim, kmatdim), dtype=tf.float64,
                                       name='K1', trainable=False)
        with tf.variable_scope("K2_matrix"):
            self.__K2 = tf.get_variable(shape=(kmatdim, kmatdim), dtype=tf.float64,
                                       name='K2', trainable=False)
        
        # Neural network approximation of dictionary elements
        with tf.variable_scope("Model", reuse=None, 
                               initializer=tf.random_uniform_initializer(maxval=std,
                                                                         minval=-std)):
            self.__psiNNx1 = self.__psiNN(self.__x1)
        with tf.variable_scope("Model", reuse=True):
            self.__psiNNy1 = self.__psiNN(self.__y1)
        with tf.variable_scope("Model", reuse=True, 
                               initializer=tf.random_uniform_initializer(maxval=std,
                                                                         minval=-std)):
            self.__psiNNx2 = self.__psiNN(self.__x2)
        with tf.variable_scope("Model", reuse=True):
            self.__psiNNy2 = self.__psiNN(self.__y2)
            
        # Loss function definition
        self.__loss_fn = tf.reduce_mean(
                        tf.square( (self.__psiNNy1 )
                      - tf.matmul(self.__psiNNx1, self.__K1))
                      + tf.square( (self.__psiNNy2 
                      - tf.matmul(self.__psiNNx2, self.__K2)) ) )

        # Optimizer
        self.__lr_val = tf.placeholder(dtype=tf.float64, shape=())
        self.__lr = tf.Variable(initial_value=init_lr, trainable=False,
                                dtype=tf.float64)
        self.__assign_lr = tf.assign(self.__lr, self.__lr_val)
        opt_D = tf.train.AdamOptimizer(learning_rate=self.__lr)
        self.__train_D = opt_D.minimize(self.__loss_fn)

        # ridge regression to find K
        self.__reg = tf.constant(shape=(),value=0.1,dtype=tf.float64)
        idmat = tf.constant(shape=(kmatdim,kmatdim),value=np.identity(kmatdim),dtype=tf.float64)
        # with tf.variable_scope("Model", reuse=True):
        #     ycurr = self.__psiNN(y)
        #     xcurr = self.__psiNN(x)
        xtx_inv = tf.matrix_inverse(self.__reg*idmat + 
                                    tf.matmul(tf.transpose(self.__psiNNx1),
                                              self.__psiNNx1))
        xty = tf.matmul(tf.transpose(self.__psiNNx1), self.__psiNNy1)
        self.__K1_reg = tf.matmul(xtx_inv, xty)
        self.__assignK1 = tf.assign(self.__K1, self.__K1_reg)
        
        # ... and K2
        xtx_inv2 = tf.matrix_inverse(self.__reg*idmat + 
                                    tf.matmul(tf.transpose(self.__psiNNx2),
                                              self.__psiNNx2))
        xty2 = tf.matmul(tf.transpose(self.__psiNNx2), self.__psiNNy2)
        
        
        
        self.__K2_reg = tf.matmul(xtx_inv2, xty2)
        self.__assignK2 = tf.assign(self.__K2, self.__K2_reg)

        # Update K1, K2 via placeholder
        self.__K1ph = tf.placeholder(tf.float64, self.__K1.get_shape(), 'K1_placeholder')
        self.__updateK1 = tf.assign(self.__K1, self.__K1ph)
        self.__K2ph = tf.placeholder(tf.float64, self.__K2.get_shape(), 'K2_placeholder')
        self.__updateK2 = tf.assign(self.__K2, self.__K2ph)        
        
        # Set Pmat
        self.__Pmat = np.random.uniform(size=(kmatdim,kmatdim))

        # Initialize session
        self.__sess = tf.Session()
        self.__sess.run(tf.global_variables_initializer())

        # Set number of parameters
        self.__num_t_params = np.sum([np.prod(v.get_shape().as_list()) 
                                     for v in tf.trainable_variables()])

        # Build complete
        print("Built tensorflow graph.")
        self.__build_flag = 0
    
    def reinitialize(self):
        assert self.__build_flag == 0, "Run build() first."
        self.__sess.run(tf.global_variables_initializer())
        print("Reinitialized all variables.")


    def dictionary(self, data, system_id=1):
        """ output dictionary """
        assert self.__build_flag == 0, "Run build() first."
        
        if len(data.shape) == 1: # handle single sample inputs
            data2d = data.reshape(1, data.size)
            if system_id == 1:
                return self.__sess.run(self.__psiNNx1,
                                       feed_dict={self.__x1: data2d}).squeeze()
            else:
                return self.__sess.run(self.__psiNNx2,
                                       feed_dict={self.__x2: data2d}).squeeze()
        else:
            if system_id == 1:
                return self.__sess.run(self.__psiNNx1,
                                       feed_dict={self.__x1: data})
            else:
                return self.__sess.run(self.__psiNNx2,
                                       feed_dict={self.__x2: data})

    def __set_K(self, x_data, y_data, system_id=1):
        """ Update K matrix from training """
        if system_id == 1:
            feed = {self.__x1: x_data, self.__y1: y_data, self.__reg: 0.01}
            return self.__sess.run(self.__K1_reg, feed_dict=feed)
        else:
            feed = {self.__x2: x_data, self.__y2: y_data, self.__reg: 0.01}
            return self.__sess.run(self.__K2_reg, feed_dict=feed)

    def __eig_decomp(self, Kval):
        """ Eigen-decomp of a matrix Kval """
        assert self.__train_flag == 0, "Run train() first."
        eigenvalues, eigenvectors = np.linalg.eig(Kval) 
        idx = eigenvalues.real.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors_inv = np.linalg.inv(eigenvectors)
        return eigenvalues, eigenvectors, eigenvectors_inv

    def train(self, num_epochs, x_data1, y_data1, x_data2, y_data2, batch_size,
              train_dict=True, verbose=True, log_interval=100, opt_interval = 10, opt_iterations = 500,
              lr_decay=0.8):
        """ Train step """

        if verbose:
            print("Started training.")
        losses = []
        assert self.__build_flag == 0, "Run build() first."        
        start_time = time.time()
        num_data = x_data1.shape[0]
        no_of_batches = num_data//batch_size
        for i in range(num_epochs):
            # Step 1: Train K_1, K_2

            # Step 1a: Get K1,K2 and set initial P
            K1mat, K2mat = self.__sess.run([self.__K1, self.__K2])
            # Pmat = np.random.uniform(size=K1mat.shape)
            if i > 0 and i % opt_interval == 0:
                
                init_mat = np.column_stack((K1mat,K2mat,self.__Pmat))

                # Step 1b: Solve constrained optimization (after network has settled a while)
                                
                A1mat, B1mat = self.__sess.run(
                    [self.__psiNNy1, self.__psiNNx1],
                    feed_dict={self.__x1: x_data1, self.__y1: y_data1})
                A2mat, B2mat = self.__sess.run( 
                    [self.__psiNNy2, self.__psiNNx2],
                    feed_dict={self.__x2: x_data2, self.__y2: y_data2})
                
                dim = K1mat.shape[0]
                constr = ({'type': 'ineq', 'fun': cons, 'args': (dim, A1mat, B1mat, A2mat, B2mat)})
                result = scipy.optimize.minimize(fp, init_mat, args=(dim, A1mat, B1mat, A2mat, B2mat),
                                                 jac=dfp, constraints=constr,
                                                 method='SLSQP', options={'disp': verbose, 'maxiter': opt_iterations})
                
                # Step 1d: Extract result and update K1, K2
                result_mat = (result.x).reshape(dim, 3*dim)
                K1mat, K2mat, self.__Pmat = (result_mat[:,:dim], result_mat[:,dim:2*dim],
                                             result_mat[:,2*dim:3*dim])
                
                self.__sess.run([self.__updateK1, self.__updateK2],
                                feed_dict={self.__K1ph: K1mat, self.__K2ph: K2mat})
                                
           
            # Step 2: Train dictionary
            if train_dict:
                ptr = 0
                for j in range(no_of_batches):
                    x_batch1, y_batch1 = x_data1[ptr:ptr+batch_size], \
                                         y_data1[ptr:ptr+batch_size]
                    x_batch2, y_batch2 = x_data2[ptr:ptr+batch_size], \
                                         y_data2[ptr:ptr+batch_size]
                    feed = {self.__x1: x_batch1, self.__y1: y_batch1,
                            self.__x2: x_batch2, self.__y2: y_batch2}
                    ptr+=batch_size
                    self.__sess.run(self.__train_D, feed_dict=feed)
            else:
                break

            # Logging
            if i%log_interval == 0:
                feed = {self.__x1: x_data1, self.__y1: y_data1,
                        self.__x2: x_data2, self.__y2: y_data2}
                losses.append([i,self.__sess.run(self.__loss_fn,
                               feed_dict=feed)])
                if verbose:
                    curr_time = time.time()
                    print("Epoch - ",str(i), \
                    " Loss - ", losses[-1][1], \
                    " LR - ", self.__sess.run(self.__lr), \
                    "Time - ", curr_time - start_time)
                    start_time = curr_time
                    
                # Adjust learning rate:
                if len(losses)>2:
                    if losses[-1][1] > losses[-2][1]:
                        print("Error increased. Decay learning rate")
                        curr_lr = self.__sess.run(self.__lr)
                        self.__sess.run(self.__assign_lr,
                                        feed_dict={self.__lr_val: lr_decay*curr_lr})
        
        # Update train flag
        self.__train_flag = 0
        
        # Set final K
        self.__Kval1 = self.__set_K(x_data1, y_data1, system_id=1)
        self.__Kval2 = self.__set_K(x_data2, y_data2, system_id=2)
        
        # Perform Eigendecomp of K 1 and 2
        self.__eigenvalues1, self.__eigenvectors1, self.__eigenvectors1_inv = self.__eig_decomp(self.__Kval1)
        self.__eigenvalues2, self.__eigenvectors2, self.__eigenvectors2_inv = self.__eig_decomp(self.__Kval2)

        # Calculate modes
        self.__modes1 = np.matmul(self.__eigenvectors1_inv, self.__B).T
        self.__modes2 = np.matmul(self.__eigenvectors2_inv, self.__B).T

        print ("Completed ", num_epochs, "epochs.")
        return losses


    def __dictNN(self, x):
        # Parameters
        dim = self.__dim
        hdim = self.__hdim
        ddim = self.__ddim
        kmatdim = ddim + 1 + dim
        num_layers = self.__num_layers
        std = 1.0 / np.sqrt(hdim)
        std_proj = 1.0 / np.sqrt(dim)
        with tf.variable_scope("Input_projection", 
                               initializer=tf.random_uniform_initializer(
                                maxval=std_proj, minval=-std_proj)):
            P = tf.get_variable(name='weights',
                                shape=(dim,hdim),
                                dtype=tf.float64)
            res_in = tf.matmul(x, P)
        with tf.variable_scope("Residual"):
            for j in range(self.__num_layers):
                layer_name = "Layer_"+str(j)
                with tf.variable_scope(layer_name):
                    W = tf.get_variable(name="weights", shape=(hdim,hdim),
                                        dtype=tf.float64)
                    b = tf.get_variable(name="biases", shape=(hdim),
                                        dtype=tf.float64)
                    if j==0: # first layer
                        res_out = res_in + self.__tf_nlr(
                            tf.matmul(res_in, W) + b)
                    else: # subsequent layers
                        res_out = res_out + self.__tf_nlr(
                            tf.matmul(res_out, W) + b)
        with tf.variable_scope("Output_projection",
                            initializer=tf.random_uniform_initializer(
                            maxval=std, minval=-std)):
            W = tf.get_variable(name="weights", shape=(hdim, ddim),
                            dtype=tf.float64)
            b = tf.get_variable(name="biases", shape=(ddim),
                            dtype=tf.float64)
            out = tf.matmul(res_out, W) + b
    #                 out = tf.nn.sigmoid(out)
    #                 out = tf.tan(TF_PI*(out-0.5))
        return out

    def __psiNN(self, data):
        """returns psi(data) where psi is approximated by a residual NN"""
        
        zout = []
        # Constant map
        zout.append(tf.ones_like(tf.slice(data,[0,0],[-1,1])))
        
        # Skip connection (identity map)
        zout.append(data)
        
        # Residual net
        zout = zout+[self.__dictNN(data)]
        
        return tf.concat(zout, axis=1)

    def eigenfunctions(self, data, system_id=1):
        """ estimated eigenfunctions """
        psix = self.dictionary(data,system_id)
        if system_id == 1:
            val = np.matmul(psix, self.__eigenvectors1)
        else:
            val = np.matmul(psix, self.__eigenvectors2)
        return val

    def predict(self, x0, traj_len, system_id=1):
        """ predict the trajectory """
        # assert x0.shape == (1, self.__dim), "Please input correct x0."
        traj = [x0.squeeze()]
        for i in range(traj_len-1):
            x_curr = traj[-1].reshape(1, self.__dim)
            efunc = self.eigenfunctions(x_curr, system_id).flatten()
            if system_id == 1:
                x_next = np.matmul(self.__modes1, self.__eigenvalues1*efunc)
            else:
                x_next = np.matmul(self.__modes2, self.__eigenvalues2*efunc)
            traj.append(x_next.real)
        return np.asarray(traj)
        
    def get_h(self, z10, z20):
        """ return the transformation function h with h(x1)=x2,
            given one matching pair h(z10)=z20. """
        assert self.__train_flag == 0, "Run train() first."  
        psi10 = self.dictionary(z10, system_id = 1)
        psi20 = self.dictionary(z20, system_id = 2)
        print('e-vals 1: ', self.__eigenvalues1)
        print('e-vals 2: ', self.__eigenvalues2)
        Dup = np.matmul(self.__eigenvectors2.T, psi20.reshape(psi20.shape[0],1))
        Ddown = np.matmul(self.__eigenvectors1.T, psi10.reshape(psi10.shape[0],1))
        D = np.diag((Dup / Ddown).flatten())
        
        def h(data):
            A = self.dictionary(data, system_id=1).T
            A = np.matmul(self.__eigenvectors1.T, A)
            A = np.matmul(D, A)
            A = np.matmul(np.linalg.inv(self.__eigenvectors2.T), A)
            A = np.matmul(self.__B.T, A)
            return np.real(A.T)
        return h

    @property
    def num_trainable_params(self):
        """ Output number of trainable parameters """
        return self.__num_t_params

    # @property
    def eigenvalues(self, system_id=1):
        """ Output eigenvalues of K """
        if system_id == 1:
            return self.__eigenvalues1
        else:
            return self.__eigenvalues2

    # @property
    def modes(self, system_id = 1):
        """ output Koopman modes """
        if system_id == 1:
            return self.__modes1
        else:
            return self.__modes2

    def get_K(self, system_id=1):
        """ output K matrix """
        assert self.__build_flag == 0, "Run build() first."
        if system_id == 1:
            return self.__Kval1
        else:
            return self.__Kval2

# QP functions
def fp(x, *args):
    """x is column stacked (K1, K2, P)"""
    d, A1, B1, A2, B2 = args
    x = x.reshape(d, 3*d)
    K1, K2, P = x[:,:d], x[:,d:2*d], x[:,2*d:3*d]
    loss1 = np.sum((A1 - B1 @ K1)**2)
    loss2 = np.sum((A2 - B2 @ K2)**2)
    
    # Enforce conjugacy between K1 and K2
    weight = 100.0
    conj_losses = weight*(np.sum((P @ K1 - K2 @ P)**2))
    return loss1 + loss2 + conj_losses

dfp = grad(fp) # jacobian

def cons(x, *args):
    """Regularization via constraints"""
    eps = 0.1
    d, A1, B1, A2, B2 = args
    x = x.reshape(d, 3*d)
    K1, K2, P = x[:,:d], x[:,d:2*d], x[:,2*d:3*d]
    return np.array([np.sum(P**2)-0.5, 5.0-np.sum(P**2), np.linalg.det(P)-0.1, 0.1-np.linalg.det(P)])
    # return np.array([np.sum(P**2)-0.5, 5.0-np.sum(P**2), np.linalg.det(P)-0.5, -np.linalg.det(P)+1.5])