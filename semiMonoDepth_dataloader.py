# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import re
import sys
import os
from glob import glob
def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])



class Dataloader(object):
    """monodepth dataloader"""

    def __init__(self, params):
        self.labeled_data_path = params.labeled_data_path
        self.unlabeled_data_path = params.unlabeled_data_path
        self.labeled_filenames_file = params.labeled_filenames_file
        self.unlabeled_filenames_file = params.unlabeled_filenames_file
        self.params = params


        self.dataset = params.dataset
        self.mode = params.mode

        self.labeled_left_image_batch = None
        self.labeled_right_image_batch = None
        self.labeled_left_disp_batch = None
        self.labeled_right_disp_batch = None
        self.unlabeled_left_image_batch = None
        self.unlabeled_right_image_batch = None


        unlabeld_input_queue = tf.train.string_input_producer([params.unlabeled_filenames_file], shuffle=False)

        unlabeled_line_reader = tf.TextLineReader()

        _, unlabeled_line = unlabeled_line_reader.read(unlabeld_input_queue)


        unlabeled_split_line = tf.string_split([unlabeled_line]).values


        unlabeled_left_image_path = tf.string_join([self.unlabeled_data_path, unlabeled_split_line[0]])
        unlabeled_right_image_path = tf.string_join([self.unlabeled_data_path, unlabeled_split_line[1]])

        unlabeled_left_image_o = self.read_image(unlabeled_left_image_path)
        unlabeled_right_image_o = self.read_image(unlabeled_right_image_path)


        if self.mode == 'train':
            labeld_input_queue = tf.train.string_input_producer([params.labeled_filenames_file], shuffle=False)
            labeled_line_reader = tf.TextLineReader()
            _, labeled_line = labeled_line_reader.read(labeld_input_queue)
            labeled_split_line = tf.string_split([labeled_line]).values

            labeled_left_image_path = tf.string_join([self.labeled_data_path, labeled_split_line[0]])
            labeled_right_image_path = tf.string_join([self.labeled_data_path, labeled_split_line[1]])
            labeled_left_disp_path = tf.string_join([self.labeled_data_path, labeled_split_line[2]])
            labeled_right_disp_path = tf.string_join([self.labeled_data_path, labeled_split_line[3]])

            labeled_left_image_o = self.read_image(labeled_left_image_path)
            labeled_right_image_o = self.read_image(labeled_right_image_path)
            labeled_left_disp = self.read_disparity(labeled_left_disp_path)
            labeled_right_disp = self.read_disparity(labeled_right_disp_path)

            # randomly augment images
            do_augment = tf.random_uniform([], 0, 1)
            labeled_left_image, labeled_right_image = tf.cond(do_augment > 0.5,
                                              lambda: self.augment_image_pair(labeled_left_image_o, labeled_right_image_o),
                                              lambda: (labeled_left_image_o, labeled_right_image_o))

            unlabeled_left_image, unlabeled_right_image = tf.cond(do_augment > 0.5,
                                                            lambda: self.augment_image_pair(unlabeled_left_image_o,
                                                                                            unlabeled_right_image_o),
                                                            lambda: (unlabeled_left_image_o, unlabeled_right_image_o))

            labeled_left_image.set_shape([None, None, 3])
            labeled_right_image.set_shape([None, None, 3])
            unlabeled_left_image.set_shape([None, None, 3])
            unlabeled_right_image.set_shape([None, None, 3])
            labeled_left_disp.set_shape([None,None,1])
            labeled_right_disp.set_shape([None, None, 1])
            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 512
            capacity = min_after_dequeue + 4 * params.batch_size

            self.labeled_left_image_batch, self.labeled_right_image_batch, self.labeled_left_disp_batch, self.labeled_right_disp_batch,self.unlabeled_left_image_batch, self.unlabeled_right_image_batch = \
                tf.train.shuffle_batch(
                [labeled_left_image, labeled_right_image, labeled_left_disp, labeled_right_disp, unlabeled_left_image, unlabeled_right_image],
                params.batch_size, capacity,
                min_after_dequeue,
                params.num_threads)

        elif self.mode == 'test':
            self.unlabeled_left_image_batch = tf.stack([unlabeled_left_image_o, tf.image.flip_left_right(unlabeled_left_image_o)], 0)
            self.unlabeled_left_image_batch.set_shape([2, None, None, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug
    def decoded_pfm(self,path):
        data,_ = self.load_pfm(open(path,'rb'))
        data /=data.shape[1]
        data = np.flip(data,axis=0)
        #data = np.flip(data, axis=1)
        data = np.expand_dims(data,2)
        return data
    def read_disparity(self,image_path):

        image = tf.py_func(self.decoded_pfm,[image_path],tf.float32)
        image.set_shape([None, None, 1])

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.image.convert_image_dtype(image, tf.float32)
        #image /= tf.shape(image)[1]
        image = tf.image.resize_images(image, [self.params.input_height, self.params.input_width], tf.image.ResizeMethod.AREA)
        #image /= self.params.width
        return image
    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path),channels=3))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image = image[:crop_height, :, :]

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.input_height, self.params.input_width], tf.image.ResizeMethod.AREA)

        return image



    '''
    Load a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    '''

    def load_pfm(self,file):
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode('utf-8').rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape), scale

    '''
    Save a Numpy array to a PFM file.
    '''

    def save_pfm(self,file, image, scale=1):
        color = None

        if image.dtype.name != 'float32':
            raise Exception('Image dtype must be float32.')

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
            color = False
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

        file.write('PF\n' if color else 'Pf\n')
        file.write('%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        file.write('%f\n' % scale)

        image.tofile(file)

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    dataloader = DepthSupervisionDataloader('/data1/disp_dataset/',None,None,None)

    left_img = sess.run(dataloader.left_image_batch)

    print(left_img)
