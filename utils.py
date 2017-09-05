################################################################################################################

import tensorflow as tf
import numpy as np
import os
import scipy.io
from dotmap import DotMap
from tqdm import tqdm
from rank_metrics import *

################################################################################################################
def get_session(gpu_num="1", gpu_fraction=0.1):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
################################################################################################################
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

################################################################################################################
class Dense(object):
    
    def __init__(self, output_dim, activation, bias=True):
        self.output_dim = output_dim
        self.activation = activation
        self.has_build = False
        self.bias = bias
        
    def build(self, input_shapes):
        input_dim = input_shapes[1]
        self.W = tf.Variable(xavier_init(input_dim, self.output_dim))
        self.b = tf.Variable(tf.zeros(self.output_dim))
        
    def __call__(self, x):
        if not self.has_build:
            shape = x.get_shape()
            shape = tuple([i.__int__() for i in shape])
            
            # Handle when the input is 1D
            if len(shape) == 1:
                self.build([0, shape[0]])
            else:
                self.build(shape)
            self.has_build = True
            
        if self.activation == 'softplus':
            transfer_fct = tf.nn.softplus
        elif self.activation == 'sigmoid':
            transfer_fct = tf.sigmoid
        elif self.activation == 'tanh':
            transfer_fct = tf.tanh
        elif self.activation == 'relu':
            transfer_fct = tf.nn.relu
        elif self.activation == 'relu6':
            transfer_fct = tf.nn.relu6
        elif self.activation == 'elu':
            transfer_fct = tf.nn.elu
        elif self.activation == 'linear':
            transfer_fct = None
        else:
            assert('Unknown activation function.')
            transfer_fct = None
        
        if self.bias == True:
            if transfer_fct is None:
                return tf.add(tf.matmul(x, self.W), self.b)
            else:
                return transfer_fct(tf.add(tf.matmul(x, self.W), self.b))
        else:
            if transfer_fct is None:
                return tf.matmul(x, self.W)
            else:
                return transfer_fct(tf.matmul(x, self.W))

################################################################################################################
class MedianHashing(object):
    
    def __init__(self):
        self.threshold = None
        self.latent_dim = None
    
    def fit(self, X):
        self.threshold = np.median(X, axis=0)
        self.latent_dim = X.shape[1]
        
    def transform(self, X):
        assert(X.shape[1] == self.latent_dim)
        binary_code = np.zeros(X.shape)
        for i in range(self.latent_dim):
            binary_code[np.nonzero(X[:,i] < self.threshold[i]),i] = 0
            binary_code[np.nonzero(X[:,i] >= self.threshold[i]),i] = 1
        return binary_code.astype(int)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

################################################################################################################
class HashNumber(object):
    ''' Represent a long integer value '''
    
    def __init__(self, bin_array):
        self.n_bits = len(bin_array)
        
        if self.n_bits > 32:
            assert(len(bin_array) % 32 == 0)

        self.bin_code = []
        for i in range(0, len(bin_array), 32):
            self.bin_code.append(self._bitarray_to_bytes(bin_array[i:i+32]))
         
    def distance(self, other):
        assert(len(self.bin_code) == len(other.bin_code))
        d = 0
        for i in range(len(self.bin_code)):
            d += self._hamming_distance(self.bin_code[i], other.bin_code[i])
        return d
    
    def _bitarray_to_bytes(self, s):
        intstr = ''.join([str(i) for i in s])
        v = int(intstr, 2)
        return v

    def _hamming_distance(self, b1, b2):
        return bin(b1^b2).count("1")  

################################################################################################################
def Load_Dataset(filename):
    dataset = scipy.io.loadmat(filename)
    x_train = dataset['train']
    x_test = dataset['test']
    x_cv = dataset['cv']
    y_train = dataset['gnd_train']
    y_test = dataset['gnd_test']
    y_cv = dataset['gnd_cv']
    
    data = DotMap()
    data.n_trains = y_train.shape[0]
    data.n_tests = y_test.shape[0]
    data.n_cv = y_cv.shape[0]
    data.n_tags = y_train.shape[1]
    data.n_feas = x_train.shape[1]

    ## Convert sparse to dense matricesimport numpy as np
    train = x_train.toarray()
    nz_indices = np.where(np.sum(train, axis=1) > 0)[0]
    train = train[nz_indices, :]
    train_len = np.sum(train > 0, axis=1)

    test = x_test.toarray()
    test_len = np.sum(test > 0, axis=1)

    cv = x_cv.toarray()
    cv_len = np.sum(cv > 0, axis=1)

    gnd_train = y_train[nz_indices, :]
    gnd_test = y_test
    gnd_cv = y_cv

    data.train = train
    data.test = test
    data.cv = cv
    data.train_len = train_len
    data.test_len = test_len
    data.cv_len = cv_len
    data.gnd_train = gnd_train
    data.gnd_test = gnd_test
    data.gnd_cv = gnd_cv
    
    return data

################################################################################################################
def run_topK_retrieval_experiment(codeTrain, codeTest, gnd_train, gnd_test, TopK=100):

    y_train = gnd_train.astype(int)
    y_test = gnd_test.astype(int)
    
    assert(codeTrain.shape[1] == codeTest.shape[1])
    assert(y_train.shape[1] == y_test.shape[1])
    assert(codeTrain.shape[0] == y_train.shape[0])
    assert(codeTest.shape[0] == y_test.shape[0])
    
    cbTrain = [HashNumber(bitarray) for bitarray in codeTrain]
    cbTest = [HashNumber(bitarray) for bitarray in codeTest]

    p_at_k = []
    avg_p = []
    ndcg_score = []
    bin_ndcg_score = []
    avg_r = []

    with tqdm(total=len(cbTest)) as pbar:
        for idx, test_bin_code in enumerate(cbTest):
            Dist = np.array([test_bin_code.distance(bincode) for bincode in cbTrain])
            TopDocIdx = np.argsort(Dist)[:TopK]

            # count number of matching labels
            num_matches = np.sum(y_test[idx] & y_train[TopDocIdx], axis=1).astype(int) 
            num_relevant_items = np.sum(np.sum(y_test[idx] & y_train, axis=1) > 0)

            relevance = (num_matches > 0).astype(int)

            # This measurement is Recall at K
            if num_relevant_items > 0:
                avg_r.append(np.sum(num_matches) / float(num_relevant_items))
            else:
                avg_r.append(0.)

            p_at_k.append(precision_at_k(relevance, TopK))
            avg_p.append(average_precision(relevance))
            bin_ndcg_score.append(ndcg_at_k(relevance, TopK))
            ndcg_score.append(ndcg_at_k(num_matches, TopK))
            
            pbar.update(1)

    avg_prec_at_k = np.mean(p_at_k)
    avg_recall_at_k = np.mean(avg_r) 
    avg_ndcg = np.mean(ndcg_score)
    print('\nPrec@K = {:.4f}, Recall@K = {:.4f}, NDCG@K = {:.4f}'.format(avg_prec_at_k, 
                                                                         avg_recall_at_k, 
                                                                         avg_ndcg))
    return avg_prec_at_k, avg_recall_at_k, avg_ndcg