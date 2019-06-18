#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import layers
from functools import partial
import numpy as np
from tqdm import tqdm
import math
import tensorflow_probability as tfp

conv2d_lrelu = partial(layers.conv2d, activation_fn=tf.nn.leaky_relu, stride=2, padding="same")
fc_relu = partial(tf.layers.dense, activation=tf.nn.relu)
z_dim = 10
beta = 4
batch = 128


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
        self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.z_sample = tf.placeholder(tf.float32, [None, z_dim])
        self.training = training
        self.dataset = []
        self.trainable = trainable

        if str(raw_input('Load_image_dataset?[yes/no] ')) == 'yes':
            self.load_data()
        # ----------------------- Net Architecture -----------------------
        # encode
        z_mean, z_logvar = self.Enc(self.img, z_dim, trainable=self.trainable, is_training= self.training)

        # sample
        if self.training:
            self.z = self.sample_from_latent_distribution(z_mean, z_logvar)
        else:
            self.z = z_mean

        # decode
        self.img_rec = self.Dec(self.z, trainable=self.trainable)

        # ----------------------- Loss Definition -----------------------

        per_sample_loss = self.make_reconstruction_loss(self.img, self.img_rec)
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


    def make_reconstruction_loss(self, true_images,
                                 reconstructed_images):
        """Wrapper that creates reconstruction loss."""
        with tf.variable_scope("reconstruction_loss"):
            per_sample_loss = tf.reduce_sum(
                tf.square(true_images - reconstructed_images), [1, 2, 3])
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
            for j in range(self.dataset.shape[0]//batch):
                i = batch * j
                _,loss = self.sess.run([self.step, self.loss], feed_dict={self.img: self.dataset[i:i+batch]})
                self.loss_list.append(float(loss))


    def image_test(self, image):
        # trainging dataset's mean and std
        mean = 116.77986
        std = 57.100357
        image = image.astype(np.float32)
        image = (image-mean)/std
        img_rec = self.sess.run(self.img_rec,
                                   feed_dict={self.img: image[np.newaxis, :]})
        print 'reconstruct image which has been processed by inverse Z-score'
        img_rec = img_rec*std + mean
        return np.squeeze(img_rec)


    def image_disentangle(self, image):
        image = image.astype(np.float32)
        z = self.sess.run(self.z,
                      feed_dict={self.img: image[np.newaxis, :]})
        print 'disentangle to vector z'
        return z


    def load_data(self):
        # for i in tqdm(range(20608)):
        #     tran_file_path = \
        #         '/home/baxter/catkin_ws/src/huang/scripts/explore_transitions/exp_transition%d.npy' \
        #         % (i + 1)
        #     transition = np.load(tran_file_path, allow_pickle=True)
        #     transition = transition.tolist()
        #     self.dataset.append(transition['observe0_img'][:,:,[2,1,0]])
        #     self.dataset.append(transition['observe1_img'][:,:,[2,1,0]])
        # self.dataset = np.array(self.dataset, dtype=np.float32)
        file_path = '/home/baxter/Documents/beta-vae/dataset.npy'
        self.dataset = np.load(file_path, allow_pickle=True)
        print "loading completed."

    def save_dataset(self):
        file_path = \
        '/home/baxter/Documents/beta-vae/dataset.npy'
        np.save(file_path, self.dataset, allow_pickle=True)


    def Save(self):
        ckpt_dir = '/home/baxter/Documents/beta-vae/checkpoints/'
        self.saver.save(self.sess, save_path=ckpt_dir+'model.ckpt')


    def load(self):
        ckpt = tf.train.get_checkpoint_state('/home/baxter/Documents/beta-vae/checkpoints/')
        self.saver.restore(self.sess, save_path=ckpt.model_checkpoint_path)


    def Enc(self, image, z_dim, trainable, is_training=True):
        bn = partial(tf.layers.batch_normalization, training=is_training)
        conv2d_lrelu_bn = partial(conv2d_lrelu, trainable=trainable, normalizer_fn=bn)
        fc_relu_TA = partial(fc_relu, trainable=trainable)
        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            image_net = conv2d_lrelu_bn(image, 32, 5)  # |inputs| num_outputs | kernel_size
            image_net = conv2d_lrelu_bn(image_net, 64, 5)
            image_net = conv2d_lrelu_bn(image_net, 128, 3)
            image_net = conv2d_lrelu_bn(image_net, 256, 3)
            image_net = conv2d_lrelu_bn(image_net, 512, 3)
            image_net = conv2d_lrelu_bn(image_net, 512, 3)
            image_net = conv2d_lrelu_bn(image_net, 512, 3)
            image_net = conv2d_lrelu_bn(image_net, 512, 3)
            image_net = layers.flatten(image_net)
            feature = bn(fc_relu_TA(image_net, z_dim))
            means = fc_relu_TA(feature, z_dim)
            log_var = fc_relu_TA(feature, z_dim)
        return means, log_var

    def Dec(self, z, trainable):
        dconv = partial(tf.layers.conv2d_transpose, strides=2, trainable=trainable,
                        activation=tf.nn.relu, padding="same")
        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):

            net = fc_relu(z, 512, trainable=trainable)
            net = tf.reshape(net, [tf.shape(z)[0], 1, 1, 512])
            net = dconv( net, 512, 3)
            net = dconv( net, 512, 3)
            net = dconv( net, 512, 3)
            net = dconv( net, 256, 3)
            net = dconv( net, 128, 3)
            net = dconv( net, 64, 3)
            net = dconv( net, 32, 5)
            image = dconv( net, 3, 5, activation=None)

        return image
