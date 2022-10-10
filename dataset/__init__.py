import os
import random

from utils.jsonencoder import NoIndent
import numpy as np
import tensorflow as tf

def augmentation(image, augment_height, augment_width, seed):
  ori_image_shape = tf.shape(image)
  image = tf.image.random_flip_left_right(image, seed=seed)
  image = tf.image.resize(image, [augment_height, augment_width])
  image = tf.image.random_crop(image, ori_image_shape, seed=seed)
  return image

def adjust_dynamic_range(images, range_in, range_out, out_dtype):
  scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
  bias = range_out[0] - range_in[0] * scale
  images = images * scale + bias
  images = tf.clip_by_value(images, range_out[0], range_out[1])
  images = tf.cast(images, dtype=out_dtype)
  return images

def preprocess_fit_train_image(images):
  images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
  return images

def postprocess_images(images):
  images = adjust_dynamic_range(images, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.dtypes.float32)
  images = tf.cast(images, dtype=tf.dtypes.uint8)
  return images
  
class Data(object):
  def __init__(self, **kwargs):

    self._dataBasePath = kwargs.pop('dataBasePath')
    self._domainList = kwargs.pop('domainList')
    self._imageShape = kwargs.pop('imageShape')
    self._doAugmentation = kwargs.pop('doAugmentation', False)

    self._numDomains = len(self._domainList)

    # generate data list
    self._images, self._imagesShuffled, self._domains = [], [], []
    for didx, domain in enumerate(self._domainList):
      dmpath = os.path.join(self._dataBasePath, domain)
      imageList = [os.path.join(dmpath, fn) for fn in os.listdir(dmpath) if os.path.splitext(fn)[-1] in [".jpg", ".png"]]
      shuffleList = random.sample(imageList, len(imageList))
      domainList = [[didx]] * len(imageList)

      self._images += imageList
      self._imagesShuffled += shuffleList
      self._domains += domainList

  def create_tf_data(self):
    return tf.data.Dataset.from_generator(self.create_data, output_types = (tf.float32, tf.float32, tf.int32))

  def create_data(self):

    def func():

      for didx in range(len(self._images)):
        
        shuffledLists = list(zip(self._images, self._imagesShuffled, self._domains))
        random.shuffle(shuffledLists)

        __images = [a0[0] for a0 in shuffledLists]
        __imagesShuffled = [a0[1] for a0 in shuffledLists]
        __domain = [a0[2] for a0 in shuffledLists]

        img0dir = __images[didx]
        img1dir = __imagesShuffled[didx]
        domain = __domain[didx]

        #img0dir = self._images[didx]
        #img1dir = self._imagesShuffled[didx]
        #domain = self._domains[didx]

        x = tf.io.read_file(img0dir)
        x_decode = tf.image.decode_jpeg(x, channels = self._imageShape[-1], dct_method = 'INTEGER_ACCURATE')
        img0 = tf.image.resize(x_decode, [self._imageShape[0], self._imageShape[1]])
        img0 = preprocess_fit_train_image(img0)

        x = tf.io.read_file(img1dir)
        x_decode = tf.image.decode_jpeg(x, channels = self._imageShape[-1], dct_method = 'INTEGER_ACCURATE')
        img1 = tf.image.resize(x_decode, [self._imageShape[0], self._imageShape[1]])
        img1 = preprocess_fit_train_image(img1)

        if self._doAugmentation:
          seed = random.randint(0, 2 ** 31 - 1)
          condition = tf.greater_equal(tf.random.uniform(shape = [], minval = 0.0, maxval = 1.0), 0.5)

          augment_height_size = self._imageShape[0] + (30 if self._imageShape[0] == 256 else int(self._imageShape[0] * 0.1))
          augment_width_size = self._imageShape[1] + (30 if self._imageShape[1] == 256 else int(self._imageShape[1] * 0.1))

          img0 = tf.cond(pred = condition,
                        true_fn = lambda : augmentation(img0, augment_height_size, augment_width_size, seed),
                        false_fn = lambda : img0)

          img1 = tf.cond(pred = condition,
                        true_fn = lambda: augmentation(img1, augment_height_size, augment_width_size, seed),
                        false_fn = lambda: img1)

        yield img0, img1, domain
    
    return func()

  @property
  def numData(self):
    return len(self._images)

  def get_log (self):
    log_json = {
      'dataBasePath': self._dataBasePath,
      'numData': self.numData,
      'domainList': self._domainList,
      
      'imageShape': NoIndent(self._imageShape),
      'doAugmentation': self._doAugmentation
    }
    return log_json