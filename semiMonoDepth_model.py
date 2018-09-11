from __future__ import absolute_import, division, print_function

from collections import namedtuple

import numpy as np

import math as m

import tensorflow as tf

import tensorflow.contrib.slim as slim

from bilinear_sampler import *

from tensorflow.contrib import rnn

import collections

Model = collections.namedtuple("Model",
                               "disp_est, "
                               "right_est, "
                               "warp_error, "
                               "ssim")


class SemidepthModel(object):
    """monodepth model"""

    def discrim_conv(self, batch_input, out_channels, stride):
        with tf.variable_scope("conv"):
            in_channels = batch_input.get_shape()[3]
            filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.02))
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
            #     => [batch, out_height, out_width, out_channels]
            padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            # print(padded_input.shape)
            # padded_input.set_shape([None,None,None,9])
            conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
            return conv

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):

        p = np.floor((kernel_size - 1) / 2).astype(np.int32)

        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])

        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):

        conv1 = self.conv(x, num_out_layers, kernel_size, 1)

        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)

        return conv2

    def upconv(self, x, num_out_layers, kernel_size, scale):

        upsample = self.upsample_nn(x, scale)

        conv = self.conv(upsample, num_out_layers, kernel_size, 1)

        return conv

    def lrelu(self, x, a):

        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(self, input):
        # done
        with tf.variable_scope("batchnorm"):
            # this block looks like it has 3 inputs on the graph unless we do this
            # input:[batch , height, width, channels]
            input = tf.identity(input)

            channels = input.get_shape()[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                                   variance_epsilon=variance_epsilon)
            return normalized

    def upsample_nn(self, x, ratio):

        s = tf.shape(x)

        h = s[1]

        w = s[2]

        # return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])
        return tf.image.resize_nearest_neighbor(x, [x.shape[1] * ratio, x.shape[2] * ratio])

    def get_disp(self, x):

        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)

        return disp
    def get_disp2(self, x):

        disp = self.conv(x, 1, 3, 1, tf.nn.sigmoid)

        return disp

    def generate_image_left(self, img, disp):

        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):

        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):

        C1 = 0.01 ** 2

        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')

        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2

        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2

        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)

        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def gradient_x(self, img):

        gx = img[:, :, :-1, :] - img[:, :, 1:, :]

        return gx

    def gradient_y(self, img):

        gy = img[:, :-1, :, :] - img[:, 1:, :, :]

        return gy

    def get_disparity_smoothness(self, disp, pyramid):

        disp_gradients_x = [self.gradient_x(d) for d in disp]

        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]

        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]

        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]

        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

        return smoothness_x + smoothness_y

    def __init__(self, params, unlabeled_left, unlabeled_right, labeled_left=None, labeled_right=None,
                 labeled_left_disp=None, labeled_right_disp=None, reuse_var=None):
        self.params = params
        self.mode = params.mode
        self.unlabeled_left = unlabeled_left
        self.unlabeled_right = unlabeled_right

        self.labeled_left = labeled_left
        self.labeled_right = labeled_right
        self.labeled_left_disp = labeled_left_disp
        self.labeled_right_disp = labeled_right_disp
        self.eps = tf.constant(1e-12, name='eps')

        self.fake_model = None
        self.real_model = None
        self.unlabeled_model = None

        self.reuse_var = reuse_var
        if self.mode == 'train':
            self.unlabeled_left_pyramid = self.scale_pyramid(self.unlabeled_left, 4)
            self.unlabeled_right_pyramid = self.scale_pyramid(self.unlabeled_right, 4)
            self.labeled_left_pyramid = self.scale_pyramid(self.labeled_left, 4)
            self.labeled_right_pyramid = self.scale_pyramid(self.labeled_right, 4)
            self.labeled_left_disp_pyramid = self.scale_pyramid(self.labeled_left_disp, 4)
            self.labeled_right_disp_pyramid = self.scale_pyramid(self.labeled_right_disp, 4)

        with tf.name_scope('unlabeled_generator'):
            with tf.variable_scope('generator', reuse=self.reuse_var):
                self.unlabeled_disp_est = self.build_generator(self.unlabeled_left)

        if self.mode == 'test':
            return

        with tf.name_scope('labeled_generator'):
            with tf.variable_scope('generator', reuse=True):
                self.labeled_disp_est = self.build_generator(self.labeled_left)

        with tf.name_scope('real_discriminaotr'):
            with tf.variable_scope('discriminator', reuse=self.reuse_var):
                self.real_model = self.build_output(self.labeled_left_disp_pyramid, self.labeled_left_pyramid,
                                                    self.labeled_right_pyramid)

                self.real_predicts = self.build_discriminator(self.real_model.warp_error)

        with tf.name_scope('fake_discriminaotr'):
            with tf.variable_scope('discriminator', reuse=True):
                self.fake_model = self.build_output(self.labeled_disp_est, self.labeled_left_pyramid,
                                                    self.labeled_right_pyramid)

                fake_inputs = None

                self.fake_predicts = self.build_discriminator(self.fake_model.warp_error)

        with tf.name_scope('unlabeled_discriminator'):
            with tf.variable_scope('discriminator', reuse=True):
                self.unlabeled_model = self.build_output(self.unlabeled_disp_est, self.unlabeled_left_pyramid,
                                                         self.unlabeled_right_pyramid)

                self.unlabeled_predicts = self.build_discriminator(self.unlabeled_model.warp_error)

        self.build_losses()
        # self.build_summaries()

    def build_losses(self):

        self.l1_losses = [tf.abs(self.unlabeled_model.warp_error[i]) for i in range(4)]
        self.ssim_losses = [tf.reduce_mean(s) for s in self.unlabeled_model.ssim]

        self.l1_reconstruction_losses = [tf.reduce_mean(l) for l in self.l1_losses]

        self.image_losses = [
            self.params.alpha_image_loss * self.ssim_losses[i] + (1 - self.params.alpha_image_loss) *
            self.l1_reconstruction_losses[i] for i in range(4)]

        self.image_loss = tf.add_n(self.image_losses)

        # DISPARITY SMOOTHNESS
        self.disp_smoothness = self.get_disparity_smoothness(self.unlabeled_model.disp_est,
                                                                  self.unlabeled_left_pyramid)

        self.disp_losses = [tf.reduce_mean(tf.abs(self.disp_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_gradient_loss = tf.add_n(self.disp_losses)

        with tf.name_scope('unlabeled_generator_loss'):
            self.unsup_gen_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss
        with tf.name_scope('labeled_generator_loss'):
            sup_l1_gen_losses = [tf.reduce_mean(tf.abs(self.labeled_disp_est[i] - self.labeled_left_disp_pyramid[i])) for i in
                                 range(4)]
            self.sup_l1_gen_loss = tf.add_n(sup_l1_gen_losses)
            sup_l2_gen_losses = [tf.reduce_mean((self.labeled_disp_est[i] - self.labeled_left_disp_pyramid[i]) ** 2) for i in
                                 range(4)]
            self.sup_l2_gen_loss = tf.add_n(sup_l2_gen_losses)
        with tf.name_scope('generator_loss'):
            self.l1_gen_loss = self.sup_l1_gen_loss + self.params.semi_weight* self.unsup_gen_loss
            self.l2_gen_loss = self.sup_l2_gen_loss + self.unsup_gen_loss
        with tf.name_scope('discriminate_loss'):
            discrim_losses = [tf.reduce_mean(
                -(tf.log(self.real_predicts[i] + self.eps) + tf.log(1 - self.fake_predicts[i] + self.eps))) for i in
                              range(4)]
            self.discrim_loss = tf.add_n(discrim_losses)
        with tf.name_scope("update_generator_loss"):
            labeled_adv_losses = [tf.reduce_mean(-tf.log(self.fake_predicts[i] + self.eps)) for i in range(4)]
            unlabeled_adv_losses = [tf.reduce_mean(-tf.log(self.unlabeled_predicts[i] + self.eps)) for i in range(4)]
            self.labeled_adv_loss = tf.add_n(labeled_adv_losses)
            self.unlabeled_adv_loss = tf.add_n(unlabeled_adv_losses)
            self.l1_update_gen_loss = self.sup_l1_gen_loss + self.params.adv_weight * self.labeled_adv_loss + self.params.adv_weight * self.unlabeled_adv_loss
            self.l2_update_gen_loss = self.sup_l2_gen_loss + self.params.adv_weight * self.labeled_adv_loss + self.params.adv_weight * self.unlabeled_adv_loss

    def build_summaries(self):
        with tf.device('/cpu:0'):
            # labeled_left, labeled_right, labeled_left_disparity, labeled_right_disparity, unlabeled left, unlabeled right,
            tf.summary.image('labeled_left', self.labeled_left)
            tf.summary.image('labeled_right', self.labeled_right)
            tf.summary.image('unlabeled_left', self.unlabeled_left)
            tf.summary.image('unlabeled_right', self.unlabeled_right)

            # # labeled_disp_est_left, labeled_disp_est_right, unlabeled_disp_est_left, unlabeled_disp_est_right, labeled_disp_left_gt, labeled_disp_right_gt
            # tf.summary.image('labeled_disp_left_est', self.fake_model.disp_left[0])
            # tf.summary.image('labeled_disp_right_est', self.fake_model.disp_right[0])
            # tf.summary.image('unlabeled_disp_left_est', self.unlabeled_model.disp_left[0])
            # tf.summary.image('unlabeled_disp_right_est', self.unlabeled_model.disp_right[0])
            # tf.summary.image('labeled_disp_left_gt', self.real_model.disp_left[0])
            # tf.summary.image('labeled_disp_right_gt', self.real_model.disp_right[0])
            # #
            # # # labeled_est_left, labeled_est_right, labeled_est_left_gt, labeled_est_right_gt, unlabeled_est_left, unlabeled_est_right
            # tf.summary.image('labeled_est_left', self.fake_model.left_est[0])
            # tf.summary.image('labeled_est_right', self.fake_model.right_est[0])
            # tf.summary.image('labeled_est_left_gt', self.real_model.left_est[0])
            # tf.summary.image('labeled_est_right_gt', self.real_model.right_est[0])
            # tf.summary.image('unlabeled_est_left', self.unlabeled_model.left_est[0])
            # tf.summary.image('unlabeled_est_right', self.unlabeled_model.right_est[0])
            #
            # # labeled_ssim_left, labeled_ssim_right, labeled_ssim_left_gt, labeled_ssim_right_gt, unlabeled_ssim_left, unlabeled_ssim_right
            # tf.summary.image('labeled_ssim_left', self.fake_model.ssim_left[0])
            # tf.summary.image('labeled_ssim_right', self.fake_model.ssim_right[0])
            # tf.summary.image('labeled_ssim_left_gt', self.real_model.ssim_left[0])
            # tf.summary.image('labeled_ssim_right_gt', self.real_model.ssim_right[0])
            # tf.summary.image('unlabeled_ssim_left', self.unlabeled_model.ssim_left[0])
            # tf.summary.image('unlabeled_ssim_right', self.unlabeled_model.ssim_right[0])
            #
            # # labeled_lr_disp_left, labeled_lr_disp_right, unlabeled_lr_disp_left, unlabeled_lr_disp_right
            # tf.summary.image('labeled_lr_disp_left', self.fake_model.right_to_left_disp[0])
            # tf.summary.image('labeled_lr_disp_right', self.fake_model.left_to_right_disp[0])
            # tf.summary.image('labeled_lr_disp_left_gt', self.real_model.right_to_left_disp[0])
            # tf.summary.image('labeled_lr_disp_right_gt', self.real_model.left_to_right_disp[0])
            # tf.summary.image('unlabeled_lr_disp_left', self.unlabeled_model.right_to_left_disp[0])
            # tf.summary.image('unlabeled_lr_disp_right', self.unlabeled_model.left_to_right_disp[0])
            #
            # # labeled_l1_left, labeled_l1_right, unlabeled_l1_left, unlabeled_l1_right
            # tf.summary.image('labeled_l1_left', self.fake_model.l1_left[0])
            # tf.summary.image('labeled_l1_right', self.fake_model.l1_right[0])
            # tf.summary.image('labeled_l1_left_gt', self.real_model.l1_left[0])
            # tf.summary.image('labeled_l1_right_gt', self.real_model.l1_right[0])
            # tf.summary.image('unlabeled_l1_left', self.unlabeled_model.l1_left[0])
            # tf.summary.image('unlabeled_l1_left', self.unlabeled_model.l1_right[0])
            #
            # # labeled_l1_lr_left, labeled_l1_lr_right, unlabeled_l1_lr_left, unlabeled_l1_lr_right
            # tf.summary.image('labeled_l1_lr_left', self.fake_model.l1_lr_left[0])
            # tf.summary.image('labeled_l1_lr_right', self.fake_model.l1_lr_right[0])
            # tf.summary.image('labeled_l1_lr_left_gt', self.real_model.l1_lr_left[0])
            # tf.summary.image('labeled_l1_lr_right_gt', self.real_model.l1_lr_right[0])
            # tf.summary.image('unlabeled_l1_lr_left', self.unlabeled_model.l1_lr_left[0])
            # tf.summary.image('unlabeled_l1_lr_left', self.unlabeled_model.l1_lr_right[0])
            #
            # # fake_predicts real_predicts,  unlabeled_predicts
            tf.summary.image('fake_predicts', self.fake_predicts[0])
            tf.summary.image('real_predicts', self.real_predicts[0])
            tf.summary.image('unlabeled_predicts', self.unlabeled_predicts[0])
            #
            # # l1_update_gen_loss l2_update_gen_loss discrim_loss labeled_adv_loss unlabeled_adv_loss
            tf.summary.scalar('l1_update_gen_loss', self.l1_update_gen_loss)
            tf.summary.scalar('l2_update_gen_loss', self.l2_update_gen_loss)
            tf.summary.scalar('discriminator_loss', self.discrim_loss)
            tf.summary.scalar('labeled_adv_loss', self.labeled_adv_loss)
            tf.summary.scalar('unlabeled_adv_loss', self.unlabeled_adv_loss)
            #
            # # l1_gen_loss l2_gen_loss self.sup_l1_gen_loss self.sup_l2_gen_loss  unsup_gen_loss
            tf.summary.scalar('l1_gen_loss', self.l1_gen_loss)
            tf.summary.scalar('l2_gen_loss', self.l2_gen_loss)
            tf.summary.scalar('sup_l1_gen_loss', self.sup_l1_gen_loss)
            tf.summary.scalar('sup_l2_gen_loss', self.sup_l2_gen_loss)
            tf.summary.scalar('unsup_gen_loss', self.unsup_gen_loss)

    def build_output(self, disp, left_pyramid, right_pyramid):

        # GENERATE IMAGES
        right_est = [self.generate_image_right(left_pyramid[i], disp[i]) for i in
                     range(4)]

        warp_error = [(right_est[i] - right_pyramid[i]) for i in range(4)]

        ssim = [self.SSIM(right_est[i], right_pyramid[i]) for i in range(4)]

        # Model = collections.namedtuple("Model",
        #                                "disp_est, "
        #                                "right_est, "
        #                                "warp_error, "
        #                                "ssim")

        return Model(disp_est=disp,
                     right_est=right_est,
                     warp_error=warp_error,
                     ssim=ssim)
    def build_discriminator(self, inputs):

        ndf = 64

        # inputs[0]: 512x256 [1]: 256 x 128 [2]: 128 x 64 [3] 64 x 32
        # inputs[0] = l1_left l1_right l1_lr_left l1_lr_right

        layers_out = []
        for i in range(4):
            with tf.variable_scope("scale_{}".format(i)):
                #                        s2@64        s2@64x2        s2@64x4         s1@64x8    => s1@64x8          s1@1
                # [0] 512x256@8  => 256x128@64   => 128x64@128   => 64x32@256    => 63x31@512    => 62x30@512    => 61x29@1
                # [1] 256x128@8  => 128x64@64    => 64x32@128    => 32x16@256    => 31x15@512    => 30x14@512    => 29x13@1
                # [2] 128x64@8   => 64x32@64     => 32x16@128    => 16x8@256     => 15x7@512     => 14x6@512     => 13x5@1
                # [3] 64x32@8    => 32x16@64     => 16x8@128     => 8x4@256      => 7x3@512      => 6x2@512      => 5x1@1

                inputs[i].set_shape([None, 256 // (2 ** i), 512 // (2 ** i), 3])
                with tf.variable_scope('layer_0'):
                    conv1 = self.discrim_conv(inputs[i], ndf, stride=2)
                    conv1_out = self.lrelu(conv1, 0.2)
                    # print("scale_{} layer_0: {}".format(i,conv1_out.shape))

                with tf.variable_scope('layer_1'):
                    conv2 = self.discrim_conv(conv1_out, ndf * 2, stride=2)
                    conv2_norm = self.batchnorm(conv2)
                    conv2_out = self.lrelu(conv2_norm, 0.2)
                    # print("scale_{} layer_1: {}".format(i,conv2_out.shape))
                with tf.variable_scope('layer_2'):
                    conv3 = self.discrim_conv(conv2_out, ndf * 4, stride=2)
                    conv3_norm = self.batchnorm(conv3)
                    conv3_out = self.lrelu(conv3_norm, 0.2)
                    # print("scale_{} layer_2: {}".format(i,conv3_out.shape))
                with tf.variable_scope('layer_3'):
                    conv4 = self.discrim_conv(conv3_out, ndf * 8, stride=1)
                    conv4_norm = self.batchnorm(conv4)
                    conv4_out = self.lrelu(conv4_norm, 0.2)
                    # print("scale_{} layer_3: {}".format(i,conv4_out.shape))
                with tf.variable_scope('layer_4'):
                    conv5 = self.discrim_conv(conv4_out, ndf * 8, stride=1)
                    conv5_norm = self.batchnorm(conv5)
                    conv5_out = self.lrelu(conv5_norm, 0.2)
                    # print("scale_{} layer_4: {}".format(i,conv5_out.shape))

                with tf.variable_scope('layer_5'):
                    conv6 = self.discrim_conv(conv5_out, 1, stride=1)
                    out = tf.sigmoid(conv6)
                    # print("scale_{} layer_5: {}".format(i,out.shape))
                layers_out.append(out)

        return layers_out

    def build_generator(self, model_input):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            if self.params.encoder == 'vgg':
                return self.build_vgg(model_input)
            elif self.params.encoder =='vgg2':
                return self.build_vgg2(model_input)
    def build_vgg2(self, model_input):
        conv = self.conv

        upconv = self.upconv

        with tf.variable_scope('encoder'):

            conv1 = self.conv_block(model_input, 32, 7)  # H/2

            conv2 = self.conv_block(conv1, 64, 5)  # H/4

            conv3 = self.conv_block(conv2, 128, 3)  # H/8

            conv4 = self.conv_block(conv3, 256, 3)  # H/16

            conv5 = self.conv_block(conv4, 512, 3)  # H/32

            conv6 = self.conv_block(conv5, 512, 3)  # H/64

            conv7 = self.conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):

            skip1 = conv1

            skip2 = conv2

            skip3 = conv3

            skip4 = conv4

            skip5 = conv5

            skip6 = conv6

        with tf.variable_scope('decoder'):

            upconv7 = upconv(conv7, 512, 3, 2)  # H/64

            concat7 = tf.concat([upconv7, skip6], 3)

            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32

            concat6 = tf.concat([upconv6, skip5], 3)

            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16

            concat5 = tf.concat([upconv5, skip4], 3)

            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8

            concat4 = tf.concat([upconv4, skip3], 3)

            iconv4 = conv(concat4, 128, 3, 1)

            self.disp4 = self.get_disp2(iconv4)

            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4

            concat3 = tf.concat([upconv3, skip2, udisp4], 3)

            iconv3 = conv(concat3, 64, 3, 1)

            self.disp3 = self.get_disp2(iconv3)

            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2

            concat2 = tf.concat([upconv2, skip1, udisp3], 3)

            iconv2 = conv(concat2, 32, 3, 1)

            self.disp2 = self.get_disp2(iconv2)

            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H

            concat1 = tf.concat([upconv1, udisp2], 3)

            iconv1 = conv(concat1, 16, 3, 1)

            self.disp1 = self.get_disp2(iconv1)

            if self.mode == 'train':
                return [self.disp1, self.disp2, self.disp3, self.disp4]
            elif self.mode == 'test':
                return self.disp1
    def build_vgg(self, model_input):

        # set convenience functions

        conv = self.conv

        upconv = self.upconv

        with tf.variable_scope('encoder'):

            conv1 = self.conv_block(model_input, 32, 7)  # H/2

            conv2 = self.conv_block(conv1, 64, 5)  # H/4

            conv3 = self.conv_block(conv2, 128, 3)  # H/8

            conv4 = self.conv_block(conv3, 256, 3)  # H/16

            conv5 = self.conv_block(conv4, 512, 3)  # H/32

            conv6 = self.conv_block(conv5, 512, 3)  # H/64

            conv7 = self.conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):

            skip1 = conv1

            skip2 = conv2

            skip3 = conv3

            skip4 = conv4

            skip5 = conv5

            skip6 = conv6

        with tf.variable_scope('decoder'):

            upconv7 = upconv(conv7, 512, 3, 2)  # H/64

            concat7 = tf.concat([upconv7, skip6], 3)

            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32

            concat6 = tf.concat([upconv6, skip5], 3)

            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16

            concat5 = tf.concat([upconv5, skip4], 3)

            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8

            concat4 = tf.concat([upconv4, skip3], 3)

            iconv4 = conv(concat4, 128, 3, 1)

            self.disp4 = self.get_disp(iconv4)

            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4

            concat3 = tf.concat([upconv3, skip2, udisp4], 3)

            iconv3 = conv(concat3, 64, 3, 1)

            self.disp3 = self.get_disp(iconv3)

            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2

            concat2 = tf.concat([upconv2, skip1, udisp3], 3)

            iconv2 = conv(concat2, 32, 3, 1)

            self.disp2 = self.get_disp(iconv2)

            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H

            concat1 = tf.concat([upconv1, udisp2], 3)

            iconv1 = conv(concat1, 16, 3, 1)

            self.disp1 = self.get_disp(iconv1)

            if self.mode == 'train':
                return [self.disp1, self.disp2, self.disp3, self.disp4]
            elif self.mode == 'test':
                return self.disp1

    def scale_pyramid(self, img, num_scales):

        scaled_imgs = [img]

        s = tf.shape(img)

        h = s[1]

        w = s[2]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)

            nh = h // ratio

            nw = w // ratio

            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))

        return scaled_imgs
