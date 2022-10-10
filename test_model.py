import os
import random
import cv2

from basemodel import BASEMODEL
from dataset import preprocess_fit_train_image
from arguments.test_args import *

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def load_images(image_path, img_size, img_channel):
  x = tf.io.read_file(image_path)
  x_decode = tf.image.decode_jpeg(x, channels=img_channel, dct_method='INTEGER_ACCURATE')
  img = tf.image.resize(x_decode, [img_size, img_size])
  img = preprocess_fit_train_image(img)

  return img

if __name__ == "__main__":

  model = BASEMODEL(False,
                    imgSize = imgSize,
                    imgCh = imgCh,
                    numDomains = numDomains,
                    numStyle = numStyle,
                    maxConvDim = maxConvDim,
                    sn = sn,
                    styleDim = styleDim,
                    hiddenDim = hiddenDim,
                    latentDim = latentDim)

  ckpt = tf.train.Checkpoint(generator = model._generator, generator_ema = model._generator_ema,
                              mappingNet = model._mappingNet, mappingNet_ema = model._mappingNet_ema,
                              styleEncoder = model._styleEncoder, styleEncoder_ema = model._styleEncoder_ema,)
  ckptManager = tf.train.CheckpointManager(ckpt, ckptPath, max_to_keep = 1)
  ckpt.restore(ckptManager.latest_checkpoint).expect_partial()
  print(f"checkpoint restored from {ckptPath}")

  if targetDomain == 0:
    srcPath = os.path.join(baseDataPath, "male")
    refPath = os.path.join(baseDataPath, "female")
  elif targetDomain == 1:
    srcPath = os.path.join(baseDataPath, "female")
    refPath = os.path.join(baseDataPath, "male")

  targetStr = f"{os.path.basename(srcPath)}2{os.path.basename(refPath)}"
  outDir = os.path.join(outPath, targetStr)
  os.makedirs(outDir, exist_ok = True)

  srcImages = [os.path.join(srcPath, fn) for fn in os.listdir(srcPath) if os.path.splitext(fn)[-1] in [".png", ".jpg"]]
  refImages = [os.path.join(refPath, fn) for fn in os.listdir(refPath) if os.path.splitext(fn)[-1] in [".png", ".jpg"]]

  for ng in range(numGenerations):
    ## reference-guided synthesis
    rdSrcImages = random.sample(srcImages, len(srcImages))
    rdRefImages = random.sample(refImages, len(refImages))

    refImg = load_images(rdRefImages[0], imgSize, imgCh)
    refImg = tf.expand_dims(refImg, axis = 0)
    refDomain = tf.expand_dims([targetDomain], axis = 0)

    srcImgs = []
    for ii in range(numSources):
      src0 = load_images(rdSrcImages[ii], imgSize, imgCh)
      srcImgs.append(tf.expand_dims(src0, axis = 0))
    srcImgs = tf.concat(srcImgs, axis = 0)

    model.refer_canvas(srcImgs, refImg, refDomain, os.path.join(outDir, f"refguided_{ng:04d}.png"), [numSources, 1], 1)
    
    ## latent-guided synthesis
    srcOuts = []
    for src0 in srcImgs:
      srcOut0 = model.latent_canvas(tf.expand_dims(src0, axis = 0), "", True, [targetDomain])
      srcOuts.append(cv2.cvtColor(srcOut0, cv2.COLOR_RGB2BGR))
    srcOuts = np.concatenate(srcOuts, axis = 1)
    cv2.imwrite(os.path.join(outDir, f"latguided_{ng:04d}.png"), srcOuts)