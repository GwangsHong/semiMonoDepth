from __future__ import absolute_import, division, print_function

# only keep warnings and errors

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from semiMonoDepth_dataloader import *
from semiMonoDepth_model import *
from average_gradients import *


parser = argparse.ArgumentParser(description='semi-supervised monodepth TensorFlow implementation.')

parser.add_argument('--mode', type=str, help='(test or train)', default='train')
parser.add_argument('--model_name', type=str, help='model name', default='semi_supervised_monodepth')
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--labeled_data_path', type=str, help='path to the labeled data')
parser.add_argument('--unlabeled_data_path', type=str, help='path to the unlabeled data', required=True)
parser.add_argument('--encoder', type=str,help = 'type of encoder, vgg or vgg2' , default='vgg2')
parser.add_argument('--labeled_filenames_file', type=str, help='path to the labeled filenames text file')
parser.add_argument('--unlabeled_filenames_file', type=str, help='path to the unlabeled filenames text file',
                    required=True)

parser.add_argument('--train_step', type=str, help='(step to train)'
                                                   '[0: semi-supervised learning]'
                                                   '[1: discriminator learning]'
                                                   '[2: generator learning]', required=True)

parser.add_argument('--input_height', type=int, help='input height', default=256)
parser.add_argument('--input_width', type=int, help='input width', default=512)
parser.add_argument('--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=100)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--lr_loss_weight', type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss', type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--adv_weight', type= float, help= 'adversarial loss weight', default= 0.01)
parser.add_argument('--semi_weight', type= float, help= 'semi-supervised loss weight', default= 0.5)
parser.add_argument('--wrap_mode', type=str, help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--num_gpus', type=int, help='number of GPUs to use for training', default=2)
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory', type=str,
                    help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory', type=str, help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain', help='if used with checkpoint_path, will restart training from step zero',
                    action='store_true')
args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def semi_sup_train(params):
    tf.reset_default_graph()
    with tf.Graph().as_default(),tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        num_training_unlabeled_samples = count_text_lines(params.unlabeled_filenames_file)
        num_training_labeled_samples = count_text_lines(params.labeled_filenames_file)
        if num_training_unlabeled_samples > num_training_labeled_samples:
            num_training_samples = num_training_unlabeled_samples
        else:
            num_training_samples = num_training_labeled_samples
        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch

        boundaries = [np.int32((3 / 5) * num_total_steps), np.int32((4 / 5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        gen_opt = tf.train.AdamOptimizer(learning_rate)

        print("total number of steps: {}".format(num_total_steps))

        dataloader = Dataloader(params)

        unalbled_left = tf.split(dataloader.unlabeled_left_image_batch, args.num_gpus, 0)
        unlabeled_right = tf.split(dataloader.unlabeled_right_image_batch, args.num_gpus, 0)
        labeled_left = tf.split(dataloader.labeled_left_image_batch, args.num_gpus, 0)
        labeled_right = tf.split(dataloader.labeled_right_image_batch, args.num_gpus, 0)
        labeled_left_disp = tf.split(dataloader.labeled_left_disp_batch, args.num_gpus, 0)
        labeled_right_disp = tf.split(dataloader.labeled_right_disp_batch, args.num_gpus, 0)

        tower_grads = []
        tower_losses = []
        reuse_var = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    model = SemidepthModel(params, unalbled_left[i], unlabeled_right[i], labeled_left[i],
                                           labeled_right[i],
                                           labeled_left_disp[i], labeled_right_disp[i], reuse_var)

                    gen_loss = model.l1_gen_loss

                    tower_losses.append(gen_loss)
                    reuse_var = True

                    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
                    gen_grads = gen_opt.compute_gradients(gen_loss, var_list=gen_tvars)
                    tower_grads.append(gen_grads)

        grads = average_gradients(tower_grads)
        gen_train = gen_opt.apply_gradients(grads, global_step=global_step)
        # tf.summary.scalar('learning_rate', learning_rate)
        # tf.summary.scalar('gen_loss', model.gen_loss)
        summary_op = tf.summary.merge_all()
        total_loss = tf.reduce_mean(tower_losses)
        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)

        #remove optimizer values to restore
        variables_to_restore_full = slim.get_variables_to_restore()
        variables_to_restore = []

        for var in variables_to_restore_full:
            if 'Adam' in var.name:
                continue
            variables_to_restore.append(var)

        train_saver = tf.train.Saver(variables_to_restore)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

                # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        print('generator train...')
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([gen_train, total_loss])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | gen loss: {:.5f} | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, training_time_left))
            if step and step % 4000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)


def discrim_train(params):
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        num_training_unlabeled_samples = count_text_lines(args.unlabeled_filenames_file)
        num_training_labeled_samples = count_text_lines(args.labeled_filenames_file)
        if num_training_unlabeled_samples > num_training_labeled_samples:
            num_training_samples = num_training_unlabeled_samples
        else:
            num_training_samples = num_training_labeled_samples
        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3 / 5) * num_total_steps), np.int32((4 / 5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        discrim_opt = tf.train.AdamOptimizer(learning_rate)

        print("total number of steps: {}".format(num_total_steps))

        dataloader = Dataloader(params)

        unalbled_left = tf.split(dataloader.unlabeled_left_image_batch, args.num_gpus, 0)
        unlabeled_right = tf.split(dataloader.unlabeled_right_image_batch, args.num_gpus, 0)
        labeled_left = tf.split(dataloader.labeled_left_image_batch, args.num_gpus, 0)
        labeled_right = tf.split(dataloader.labeled_right_image_batch, args.num_gpus, 0)
        labeled_left_disp = tf.split(dataloader.labeled_left_disp_batch, args.num_gpus, 0)
        labeled_right_disp = tf.split(dataloader.labeled_right_disp_batch, args.num_gpus, 0)

        tower_grads = []
        tower_losses = []
        reuse_var = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    model = SemidepthModel(params, unalbled_left[i], unlabeled_right[i], labeled_left[i], labeled_right[i],
                                           labeled_left_disp[i], labeled_right_disp[i], reuse_var)

                    discrim_loss = model.discrim_loss

                    tower_losses.append(discrim_loss)
                    reuse_var = True

                    discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
                    discrim_grads = discrim_opt.compute_gradients(discrim_loss, var_list=discrim_tvars)
                    tower_grads.append(discrim_grads)

        grads = average_gradients(tower_grads)
        discrim_train = discrim_opt.apply_gradients(grads, global_step=global_step)

        # tf.summary.scalar('learning_rate', learning_rate)
        # tf.summary.scalar('gen_loss', model.gen_loss)
        summary_op = tf.summary.merge_all()
        total_loss = tf.reduce_mean(tower_losses)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        variables_to_restore_full = slim.get_variables_to_restore()
        variables_to_restore = []

        for var in variables_to_restore_full:
            if 'Adam' in var.name:
                continue
            variables_to_restore.append(var)

        train_saver = tf.train.Saver(variables_to_restore)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)




        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        print('discriminator train...')
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([discrim_train, total_loss])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | discrim_loss: {:.5f} | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, training_time_left))
            if step and step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 2000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def gen_train(params):
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_unlabeled_samples = count_text_lines(args.unlabeled_filenames_file)
        num_training_labeled_samples = count_text_lines(args.labeled_filenames_file)
        if num_training_unlabeled_samples > num_training_labeled_samples:
            num_training_samples = num_training_unlabeled_samples
        else:
            num_training_samples = num_training_labeled_samples
        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        boundaries = [np.int32((3 / 5) * num_total_steps), np.int32((4 / 5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        gen_opt = tf.train.AdamOptimizer(learning_rate)

        print("total number of steps: {}".format(num_total_steps))

        dataloader = Dataloader(params)

        unalbled_left = tf.split(dataloader.unlabeled_left_image_batch, args.num_gpus, 0)
        unlabeled_right = tf.split(dataloader.unlabeled_right_image_batch, args.num_gpus, 0)
        labeled_left = tf.split(dataloader.labeled_left_image_batch, args.num_gpus, 0)
        labeled_right = tf.split(dataloader.labeled_right_image_batch, args.num_gpus, 0)
        labeled_left_disp = tf.split(dataloader.labeled_left_disp_batch, args.num_gpus, 0)
        labeled_right_disp = tf.split(dataloader.labeled_right_disp_batch, args.num_gpus, 0)

        tower_grads = []
        tower_losses = []
        reuse_var = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d' % i):
                    model = SemidepthModel(params, unalbled_left[i], unlabeled_right[i], labeled_left[i], labeled_right[i],
                                           labeled_left_disp[i], labeled_right_disp[i], reuse_var)

                    update_gen_loss = model.l1_update_gen_loss

                    tower_losses.append(update_gen_loss)
                    reuse_var = True

                    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
                    gen_grads = gen_opt.compute_gradients(update_gen_loss, var_list=gen_tvars)
                    tower_grads.append(gen_grads)

        grads = average_gradients(tower_grads)
        gen_train = gen_opt.apply_gradients(grads, global_step=global_step)
        # tf.summary.scalar('learning_rate', learning_rate)
        # tf.summary.scalar('gen_loss', model.gen_loss)
        summary_op = tf.summary.merge_all()
        total_loss = tf.reduce_mean(tower_losses)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        variables_to_restore_full = slim.get_variables_to_restore()
        variables_to_restore = []

        for var in variables_to_restore_full:
            if 'Adam' in var.name:
                continue
            variables_to_restore.append(var)

        train_saver = tf.train.Saver(variables_to_restore)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        print('update generator train...')
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value = sess.run([gen_train, total_loss])
            duration = time.time() - before_op_time
            if step and step % 100 == 0:
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | update_gen_loss: {:.5f} | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, training_time_left))
            if step and step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 2000 == 0:# OPTIMIZER
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = Dataloader(params)
    left = dataloader.unlabeled_left_image_batch
    right = dataloader.unlabeled_right_image_batch


    model = SemidepthModel(params, left, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.reset_default_graph()
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.unlabeled_filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities = np.zeros((num_test_samples, params.input_height, params.input_width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.input_height, params.input_width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.unlabeled_disp_est)
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy', disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)

    print('done.')

def main(_):
    # semi_sup_train: train generator (semi-supervision)
    # discrim_train: frozen generator train discriminator
    # gen_train: frozen discriminator train generator
    if args.mode == 'train':
        if args.train_step == '0':
            semi_sup_train(args)
        elif args.train_step == '1':
            discrim_train(args)
        elif args.train_step == '2':
            gen_train(args)
    elif args.mode == 'test':
        test(args)




if __name__ == '__main__':
    tf.app.run()
