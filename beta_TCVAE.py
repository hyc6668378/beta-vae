#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import layers
from functools import partial
import numpy as np
from tqdm import tqdm
import math

conv2d_lrelu = partial(layers.conv2d, activation_fn=tf.nn.leaky_relu, stride=2, padding="same")
fc_relu = partial(tf.layers.dense, activation=tf.nn.relu)
z_dim = 8
beta = 4
batch = 64


def gaussian_log_density(samples, mean, log_var):
    pi = tf.constant(math.pi)
    normalization = tf.log(2. * pi)
    inv_sigma = tf.exp(-log_var)
    tmp = (samples - mean)
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

class beta_TCVAE():
    def __init__(self,training = True, lr=0.0008, trainable= True):
        # input
        self.sess = tf.InteractiveSession()
        self.loss_list= []
        self.head_img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.right_img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.left_img = tf.placeholder(tf.float32, [None, 256, 256, 3])

        self.z_sample = tf.placeholder(tf.float32, [None, z_dim])
        self.training = training
        self.dataset=np.zeros([1,256,256,3], dtype=np.uint8) # temp
        self.trainable = trainable
        self.mean = 110.6034
        self.std = 47.03846

        # ----------------------- Net Architecture -----------------------
        # encode
        z_mean, z_logvar = self.Enc(self.head_img, self.right_img, self.left_img, z_dim,
                                    trainable=self.trainable, is_training= self.training)

        # sample
        if self.training:
            self.z = self.sample_from_latent_distribution(z_mean, z_logvar)
        else:
            self.z = z_mean

        # decode
        self.dec_head, self.dec_right, self.dec_left = self.Dec(self.z, trainable=self.trainable)

        # ----------------------- Loss Definition -----------------------

        per_sample_loss = self.make_reconstruction_loss(self.head_img, self.right_img, self.left_img,
                                                        self.dec_head, self.dec_right, self.dec_left)
        reconstruction_loss = tf.reduce_mean(per_sample_loss)
        kl_loss = self.compute_gaussian_kl(z_mean, z_logvar)
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, self.z)
        self.loss = tf.add(reconstruction_loss, regularizer, name="loss")

        # ----------------------- Optimisation Setting -----------------------

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2 = 0.999).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        tf.summary.scalar('loss', self.loss)
        self.merge_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('train/', self.sess.graph)

    def make_reconstruction_loss(self, true_head_img, true_right_img, true_left_img,
                                            dec_head, dec_right, dec_left):
        """Wrapper that creates reconstruction loss."""
        with tf.variable_scope("reconstruction_loss"):
            loss = tf.square(true_head_img - dec_head)
            loss += tf.square(true_right_img - dec_right)
            loss += tf.square(true_left_img - dec_left)
            per_sample_loss = tf.reduce_sum(loss, [1, 2, 3])

        return per_sample_loss


    def compute_gaussian_kl(self, z_mean, z_logvar):
        """Compute KL divergence between input Gaussian and Standard Normal."""
        return tf.reduce_mean(
            0.5 * tf.reduce_sum(
                tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]),
            name="kl_loss")


    def total_correlation(self, z, z_mean, z_logvar):
        """Estimate of total correlation on a batch.
        We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
        log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
        for the minimization. The constant should be equal to (num_latents - 1) *
        log(batch_size * dataset_size)
        Args:
          z: [batch_size, num_latents]-tensor with sampled representation.
          z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
          z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
        Returns:
          Total correlation estimated on a batch.
        """
        # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
        # tensor of size [batch_size, batch_size, num_latents]. In the following
        # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
        log_qz_prob = gaussian_log_density(
            tf.expand_dims(z, 1), tf.expand_dims(z_mean, 0),
            tf.expand_dims(z_logvar, 0))
        # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
        # + constant) for each sample in the batch, which is a vector of size
        # [batch_size,].
        log_qz_product = tf.reduce_sum(
            tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
            axis=1,
            keepdims=False)
        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
        # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
        log_qz = tf.reduce_logsumexp(
            tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
            axis=1,
            keepdims=False)
        return tf.reduce_mean(log_qz - log_qz_product)


    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        tc = (beta - 1.) * self.total_correlation(z_sampled, z_mean, z_logvar)
        return tc + kl_loss


    def sample_from_latent_distribution(self, z_mean, z_logvar):
        """Samples from the Gaussian distribution defined by z_mean and z_logvar."""
        return tf.add(
            z_mean,
            tf.exp(z_logvar / 2) * tf.random_normal(tf.shape(z_mean), 0, 1),
            name="sampled_latent_variable")


    def fit(self, epoch=50):
        for _ in tqdm(range(epoch)):
            for j in range(self.dataset.shape[0]//(batch*3)):
                i = batch*3 * j
                _,train_summary = self.sess.run([self.step, self.merge_summary],
                                                feed_dict={self.head_img: self.dataset[i  :i+batch*3  : 3],
                                                           self.right_img:self.dataset[i+1:i+1+batch*3: 3],
                                                           self.left_img: self.dataset[i+2:i+2+batch*3: 3]})
                self.train_writer.add_summary(train_summary, j)

    def image_preproces(self,head_img, right_img, left_img):

        head_img = head_img.astype(np.float32)
        right_img = right_img.astype(np.float32)
        left_img = left_img.astype(np.float32)

        head_img = (head_img - self.mean) / self.std
        right_img = (right_img - self.mean) / self.std
        left_img = (left_img - self.mean) / self.std

        return head_img, right_img, left_img

    def de_Z_score(self, head_img, right_img, left_img):
        head_img = head_img * self.std + self.mean
        right_img = right_img * self.std + self.mean
        left_img = left_img * self.std + self.mean
        return head_img, right_img, left_img

    def image_test(self, head_img, right_img, left_img):
        # trainging dataset's mean and std

        head_img, right_img, left_img = self.image_preproces(head_img, right_img, left_img)

        dec_head, dec_right, dec_left = self.sess.run([self.dec_head, self.dec_right, self.dec_left],
                                           feed_dict={self.head_img:  head_img[np.newaxis, :],
                                                      self.right_img: right_img[np.newaxis, :],
                                                      self.left_img:  left_img[np.newaxis, :]})
        print 'reconstruct image which has been processed by inverse Z-score'
        dec_head, dec_right, dec_left = self.de_Z_score(dec_head, dec_right, dec_left)
        return np.squeeze(dec_head), np.squeeze(dec_right), np.squeeze(dec_left)


    def image_disentangle(self, head_img, right_img, left_img):
        head_img, right_img, left_img = self.image_preproces(head_img, right_img, left_img)
        z = self.sess.run(self.z,
                          feed_dict={self.head_img:  head_img[np.newaxis, :],
                                     self.right_img: right_img[np.newaxis, :],
                                     self.left_img:  left_img[np.newaxis, :]})
        print 'disentangle to vector z'
        return z


    def load_data(self, data_id):
        for i in range(5):
            tran_file_path = \
                'dataset/dataset%d.npy' \
                % (data_id+i)
            self.dataset = np.concatenate((self.dataset, np.load(tran_file_path, allow_pickle=True)), axis=0)
        self.dataset = np.delete(self.dataset, 0, 0)  # delete temp raw
        self.dataset = self.dataset.astype(np.float32)
        # z-score
        self.dataset = (self.dataset - self.mean)/self.std
        print "loading completed."

    def Save(self):
        ckpt_dir = '/home/baxter/Documents/beta-vae/checkpoints/'
        self.saver.save(self.sess, save_path=ckpt_dir+'model.ckpt')


    def load(self):
        ckpt = tf.train.get_checkpoint_state('/home/baxter/Documents/beta-vae/checkpoints/')
        self.saver.restore(self.sess, save_path=ckpt.model_checkpoint_path)


    def Enc(self, head_img, right_img, left_img, z_dim, trainable, is_training=True):
        bn = partial(tf.layers.batch_normalization, training=is_training)
        conv2d_lrelu_bn = partial(conv2d_lrelu, trainable=trainable, normalizer_fn=bn)
        fc_relu_TA = partial(fc_relu, trainable=trainable)
        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('head', reuse=tf.AUTO_REUSE):
                head = conv2d_lrelu_bn(head_img, 32, 5)  # |inputs| num_outputs | kernel_size
                head = conv2d_lrelu_bn(head, 64, 5)
                head = conv2d_lrelu_bn(head, 128, 3)
                head = conv2d_lrelu_bn(head, 256, 3)
                head = conv2d_lrelu_bn(head, 512, 3)
                head = conv2d_lrelu_bn(head, 512, 3)
                head = conv2d_lrelu_bn(head, 512, 3)
                head = conv2d_lrelu_bn(head, 512, 3)
                head = layers.flatten(head)
                feature_head = bn(fc_relu_TA(head, z_dim))
            with tf.variable_scope('right', reuse=tf.AUTO_REUSE):
                right = conv2d_lrelu_bn(right_img, 32, 5)  # |inputs| num_outputs | kernel_size
                right = conv2d_lrelu_bn(right, 64, 5)
                right = conv2d_lrelu_bn(right, 128, 3)
                right = conv2d_lrelu_bn(right, 256, 3)
                right = conv2d_lrelu_bn(right, 512, 3)
                right = conv2d_lrelu_bn(right, 512, 3)
                right = conv2d_lrelu_bn(right, 512, 3)
                right = conv2d_lrelu_bn(right, 512, 3)
                right = layers.flatten(right)
                feature_right = bn(fc_relu_TA(right, z_dim))
            with tf.variable_scope('left', reuse=tf.AUTO_REUSE):
                left = conv2d_lrelu_bn(left_img, 32, 5)  # |inputs| num_outputs | kernel_size
                left = conv2d_lrelu_bn(left, 64, 5)
                left = conv2d_lrelu_bn(left, 128, 3)
                left = conv2d_lrelu_bn(left, 256, 3)
                left = conv2d_lrelu_bn(left, 512, 3)
                left = conv2d_lrelu_bn(left, 512, 3)
                left = conv2d_lrelu_bn(left, 512, 3)
                left = conv2d_lrelu_bn(left, 512, 3)
                left = layers.flatten(left)
                feature_left = bn(fc_relu_TA(left, z_dim))
            feature = tf.concat([feature_head, feature_right, feature_left], axis=1)
            feature = fc_relu_TA(feature, z_dim)
            means = fc_relu_TA(feature, z_dim)
            log_var = fc_relu_TA(feature, z_dim)
        return means, log_var

    def Dec(self, z, trainable):
        dconv = partial(tf.layers.conv2d_transpose, strides=2, trainable=trainable,
                        activation=tf.nn.relu, padding="same")
        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('head', reuse=tf.AUTO_REUSE):
                dec_head = fc_relu(z, 512, trainable=trainable)
                dec_head = tf.reshape(dec_head, [tf.shape(z)[0], 1, 1, 512])
                dec_head = dconv( dec_head, 512, 3)
                dec_head = dconv( dec_head, 512, 3)
                dec_head = dconv( dec_head, 512, 3)
                dec_head = dconv( dec_head, 256, 3)
                dec_head = dconv( dec_head, 128, 3)
                dec_head = dconv( dec_head, 64, 3)
                dec_head = dconv( dec_head, 32, 5)
                dec_head = dconv( dec_head, 3, 5, activation=None)
            with tf.variable_scope('right', reuse=tf.AUTO_REUSE):
                dec_right = fc_relu(z, 512, trainable=trainable)
                dec_right = tf.reshape(dec_right, [tf.shape(z)[0], 1, 1, 512])
                dec_right = dconv(dec_right, 512, 3)
                dec_right = dconv(dec_right, 512, 3)
                dec_right = dconv(dec_right, 512, 3)
                dec_right = dconv(dec_right, 256, 3)
                dec_right = dconv(dec_right, 128, 3)
                dec_right = dconv(dec_right, 64, 3)
                dec_right = dconv(dec_right, 32, 5)
                dec_right = dconv(dec_right, 3, 5, activation=None)
            with tf.variable_scope('left', reuse=tf.AUTO_REUSE):
                dec_left = fc_relu(z, 512, trainable=trainable)
                dec_left = tf.reshape(dec_left, [tf.shape(z)[0], 1, 1, 512])
                dec_left = dconv(dec_left, 512, 3)
                dec_left = dconv(dec_left, 512, 3)
                dec_left = dconv(dec_left, 512, 3)
                dec_left = dconv(dec_left, 256, 3)
                dec_left = dconv(dec_left, 128, 3)
                dec_left = dconv(dec_left, 64, 3)
                dec_left = dconv(dec_left, 32, 5)
                dec_left = dconv(dec_left, 3, 5, activation=None)

        return dec_head, dec_right, dec_left
