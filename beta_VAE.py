#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import layers
from functools import partial
import numpy as np
from tqdm import tqdm

conv2d_lrelu = partial(layers.conv2d, activation_fn=tf.nn.leaky_relu, stride=2, padding='valid')
fc_relu = partial(tf.layers.dense, activation=tf.nn.relu)
z_dim = 32
beta = 0.1
lr = 0.0002
batch = 64

class beta_VAE():
    def __init__(self):
        # input
        self.sess = tf.InteractiveSession()

        self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.z_sample = tf.placeholder(tf.float32, [None, z_dim])
        self.training = True

        self.dataset = []

        if str(raw_input('Load_image_dataset?[yes/no] ')) == 'yes':
            self.load_data()

        # encode
        z_mu, z_log_sigma_sq = self.Enc(self.img, z_dim, is_training= self.training)

        # sample
        epsilon = tf.random_normal(tf.shape(z_mu))
        if self.training:
            self.z = z_mu + tf.exp(0.5 * z_log_sigma_sq) * epsilon
        else:
            self.z = z_mu

        # decode
        self.img_rec = self.Dec(self.z, is_training=self.training)

        # loss
        rec_loss = tf.losses.mean_squared_error(self.img, self.img_rec)
        kld_loss = -tf.reduce_mean(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))
        loss = rec_loss + kld_loss * beta

        # otpim
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(loss)

        self.sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)

    def fit(self, epoch=50):
        for i in tqdm(range(epoch)):
            for j in range(self.dataset.shape[0]//batch):
                self.sess.run(self.step, feed_dict={self.img: self.dataset[j:j+batch]})

    def image_test(self, image):
        image = image.astype(np.float32)
        z, img_rec = self.sess.run([self.z, self.img_rec],
                                   feed_dict={self.img: image[np.newaxis, :]})
        print 'return (disentangle vector z) and (reconstructed image):'
        return z, np.squeeze(img_rec)

    def load_data(self):
        self.dataset = []
        for i in tqdm(range(20608)):
            tran_file_path = \
                '/home/baxter/catkin_ws/src/huang/scripts/explore_transitions/exp_transition%d.npy' \
                % (i + 1)
            transition = np.load(tran_file_path, allow_pickle=True)
            transition = transition.tolist()
            self.dataset.append(transition['observe0_img'][:,:,[2,1,0]])
            self.dataset.append(transition['observe1_img'][:,:,[2,1,0]])
        self.dataset = np.array(self.dataset, dtype=np.float32)
        print "loading completed."

    def Save(self):
        ckpt_dir = '/home/baxter/Documents/beta-vae/checkpoints/'
        self.saver.save(self.sess, save_path=ckpt_dir+'model.ckpt')

    def load(self):
        ckpt = tf.train.get_checkpoint_state('/home/baxter/Documents/beta-vae/checkpoints/')
        self.saver.restore(self.sess, save_path=ckpt.model_checkpoint_path)

    def Enc(self, image, z_dim, is_training=True):
        bn = partial(tf.layers.batch_normalization, training=is_training)
        conv2d_lrelu_bn = partial(conv2d_lrelu, normalizer_fn=bn)
        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            image_net = conv2d_lrelu_bn(image, 32, 5)  # |inputs| num_outputs | kernel_size
            image_net = conv2d_lrelu_bn(image_net, 64, 5)
            image_net = conv2d_lrelu_bn(image_net, 128, 3)
            image_net = conv2d_lrelu_bn(image_net, 256, 3)
            image_net = conv2d_lrelu_bn(image_net, 512, 3)
            image_net = conv2d_lrelu_bn(image_net, 512, 6)
            image_net = layers.flatten(image_net)
            feature = bn(fc_relu(image_net, z_dim))
            z_mu = fc_relu(feature, z_dim)
            z_log_sigma_sq = fc_relu(feature, z_dim)
        return z_mu, z_log_sigma_sq


    def Dec(self, z, is_training=True):
        dconv = partial(tf.nn.conv2d_transpose, strides=[1, 2, 2, 1], padding="VALID")
        relu = partial(tf.nn.relu)
        bn = partial(tf.layers.batch_normalization, training=is_training)
        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            kernel1 = tf.random_normal(shape=[6, 6, 512, 512])
            kernel2 = tf.random_normal(shape=[3, 3, 256, 512])
            kernel3 = tf.random_normal(shape=[3, 3, 128, 256])
            kernel4 = tf.random_normal(shape=[3, 3, 64, 128])
            kernel5 = tf.random_normal(shape=[5, 5, 32, 64])
            kernel6 = tf.random_normal(shape=[5, 5, 3, 32])

            net = fc_relu(z, 512)
            net = tf.reshape(net, [-1, 1, 1, 512])
            net = relu(bn(dconv(net, kernel1, [tf.shape(z)[0], 6, 6, 512])))
            net = relu(bn(dconv(net, kernel2, [tf.shape(z)[0], 14, 14, 256])))
            net = relu(bn(dconv(net, kernel3, [tf.shape(z)[0], 30, 30, 128])))
            net = relu(bn(dconv(net, kernel4, [tf.shape(z)[0], 61, 61, 64])))
            net = relu(bn(dconv(net, kernel5, [tf.shape(z)[0], 126, 126, 32])))
            image = relu(bn(dconv(net, kernel6, [tf.shape(z)[0], 256, 256, 3])))

        return image