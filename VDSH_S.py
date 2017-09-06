import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils import *

####################################################################################################################
## Unsupervised Learning model
####################################################################################################################

class VDSH_S(object):
    def __init__(self, sess, latent_dim, n_feas, n_tags, use_cross_entropy=True):
        self.sess = sess
        self.n_feas = n_feas
        
        self.latent_dim = latent_dim
        
        self.n_tags = n_tags
        
        n_batches = 1
        self.n_batches = n_batches

        self.use_cross_entropy = use_cross_entropy
        
        self.hidden_dim = 500
        self.build()
    
    def transform(self, docs):
        z_data = []
        for i in tqdm(range(len(docs))):
            doc = docs[i]
            word_indice = np.where(doc > 0)[0]
            z = self.sess.run(self.z_mean, 
                           feed_dict={ self.input_bow: doc.reshape((-1, self.n_feas)),
                                       self.input_bow_idx: word_indice,
                                       self.keep_prob: 1.0})
            z_data.append(z[0])
        return z_data

    def calc_reconstr_error(self):
        # Pick score for those visiable words
        p_x_i_scores0 = tf.gather(self.p_x_i, self.input_bow_idx)
        weight_scores0 = tf.gather(tf.squeeze(self.input_bow), self.input_bow_idx)
        return -tf.reduce_sum(tf.log(tf.maximum(p_x_i_scores0 * weight_scores0, 1e-10)))

    def calc_KL_loss(self):
            return -0.5 * tf.reduce_sum(tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) 
                                                      - tf.exp(self.z_log_var), axis=1))
    
    def calc_Pred_loss(self):
        if self.use_cross_entropy:
            return tf.reduce_sum(tf.pow(self.tag_prob - self.labels, 2), axis=1)
        else:
            return -tf.reduce_sum(self.labels * tf.log(tf.maximum(self.tag_prob,1e-10))
                           + (1-self.labels) * tf.log(tf.maximum(1 - self.tag_prob, 1e-10)), axis=1)
    
    def build(self):
        # BOW
        self.input_bow = tf.placeholder(tf.float32, [1, self.n_feas], name="Input_BOW")
        # indices
        self.input_bow_idx = tf.placeholder(tf.int32, [None], name="Input_bow_Idx")
        # labels
        self.labels = tf.placeholder(tf.float32, [1, self.n_tags], name="Input_Labels")
        
        self.kl_weight = tf.placeholder(tf.float32, name="KL_Weight")
        self.keep_prob = tf.placeholder(tf.float32, name="KL_Weight")

        ## Inference network q(z|x)
        self.z_enc_1 = Dense(self.hidden_dim, activation='relu')(self.input_bow)
        self.z_enc_2 = Dense(self.hidden_dim, activation='relu')(self.z_enc_1)
        self.z_enc_3 = tf.nn.dropout(self.z_enc_2, keep_prob=self.keep_prob)
        
        self.z_mean = Dense(self.latent_dim, activation='linear')(self.z_enc_3)
        self.z_log_var = Dense(self.latent_dim, activation='sigmoid')(self.z_enc_3)
        
        # Sampling Layers X
        self.eps_z = tf.random_normal((self.n_batches, self.latent_dim), 0, 1, dtype=tf.float32)
        self.z_sample = self.z_mean + tf.sqrt(tf.exp(self.z_log_var)) * self.eps_z
        
        # Decoding Layers
        self.R = tf.Variable(tf.random_normal([self.n_feas, self.latent_dim]), name="R_Mat")
        self.b = tf.Variable(tf.zeros([self.n_feas]), name="B_Mat")
        self.e = -tf.matmul(self.z_sample, self.R, transpose_b=True) + self.b
        self.p_x_i = tf.squeeze(tf.nn.softmax(self.e))
        
        self.tag_prob = Dense(self.n_tags, activation='sigmoid')(self.z_sample)
        self.pred_loss = self.calc_Pred_loss()
        
        self.reconstr_err = self.calc_reconstr_error()
        self.kl_loss = self.calc_KL_loss()
        
        self.cost = self.reconstr_err + self.kl_weight * self.kl_loss + self.pred_loss