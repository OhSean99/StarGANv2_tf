import copy
from pydoc import isdata
from statistics import mode
import tqdm
import json
import os
import time
import logging
import absl.logging

import tensorflow as tf
import numpy as np
import PIL.Image

from networks.net_bank import GENERATOR, MAPPINGNETWORK, STYLEENCODER, DISCRIMINATOR
from networks.loss_bank import *
from utils.jsonencoder import JsonEncoder
from dataset import postprocess_images, preprocess_fit_train_image

def get_logger(loggingName, loggingFolder):
  logging.root.removeHandler(absl.logging._absl_handler)
  absl.logging._warn_preinit_stderr = False
  log_formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')

  logger = logging.getLogger()

  console_handler = logging.StreamHandler()
  console_handler.setFormatter(log_formatter)
  logger.addHandler(console_handler)

  file_handler = logging.FileHandler(os.path.join(loggingFolder, f'{loggingName}.log'))
  file_handler.setFormatter(log_formatter)
  logger.addHandler(file_handler)
  logger.setLevel(logging.INFO)

  return logger

class BASEMODEL(object):
  def __init__(self, istrain, **kwargs):
    
    self._imgSize = kwargs.pop("imgSize", 256)
    self._imgCh = kwargs.pop("imgCh", 3)
    self._numDomains = kwargs.pop("numDomains", 2)
    self._numStyle = kwargs.pop("numStyle", 5)
    self._maxConvDim = kwargs.pop("maxConvDim", 512)
    self._sn = kwargs.pop("sn", False)
    self._styleDim = kwargs.pop("styleDim", 64)
    self._hiddenDim = kwargs.pop("hiddenDim", 512)
    self._latentDim = kwargs.pop("latentDim", 16)

    self._generator = GENERATOR(imgSize = self._imgSize, imgCh = self._imgCh,
                                styleDim = self._styleDim, maxConvDim = self._maxConvDim, 
                                sn = self._sn)
    self._mappingNet = MAPPINGNETWORK(styleDim = self._styleDim, hiddenDim = self._hiddenDim,
                                      numDomains = self._numDomains, sn = self._sn)
    self._styleEncoder = STYLEENCODER(imgSize = self._imgSize, styleDim = self._styleDim,
                                      numDomains = self._numDomains, maxConvDim = self._maxConvDim,
                                      sn = self._sn)
    if istrain:
      self._discriminator = DISCRIMINATOR(imgSize = self._imgSize, numDomains = self._numDomains,
                                          maxConvDim = self._maxConvDim, sn = self._sn)

    self._generator_ema = copy.deepcopy(self._generator)
    self._mappingNet_ema = copy.deepcopy(self._mappingNet)
    self._styleEncoder_ema = copy.deepcopy(self._styleEncoder)

    # build
    _ = self._generator([tf.convert_to_tensor(np.ones((1, self._imgSize, self._imgSize, self._imgCh), np.float32)),
                        tf.convert_to_tensor(np.ones(shape=[1, self._styleDim], dtype = np.float32))])
    _ = self._generator_ema([tf.convert_to_tensor(np.ones((1, self._imgSize, self._imgSize, self._imgCh), np.float32)),
                            tf.convert_to_tensor(np.ones(shape=[1, self._styleDim], dtype = np.float32))])
    self._generator.summary()

    _ = self._mappingNet([tf.convert_to_tensor(np.ones((1, self._latentDim), np.float32)),
                        tf.convert_to_tensor(np.ones(shape=[1, 1], dtype = np.int32))])
    _ = self._mappingNet_ema([tf.convert_to_tensor(np.ones((1, self._latentDim), np.float32)),
                        tf.convert_to_tensor(np.ones(shape=[1, 1], dtype = np.int32))])
    self._mappingNet.summary()
                      
    _ = self._styleEncoder([tf.convert_to_tensor(np.ones((1, self._generator._imgSize, self._generator._imgSize, self._generator._imgCh), np.float32)),
                            tf.convert_to_tensor(np.ones(shape=[1, 1], dtype = np.int32))])
    _ = self._styleEncoder_ema([tf.convert_to_tensor(np.ones((1, self._generator._imgSize, self._generator._imgSize, self._generator._imgCh), np.float32)),
                            tf.convert_to_tensor(np.ones(shape=[1, 1], dtype = np.int32))])
    self._styleEncoder.summary()

    if istrain:
      _ = self._discriminator([tf.convert_to_tensor(np.ones((1, self._generator._imgSize, self._generator._imgSize, self._generator._imgCh), np.float32)),
                              tf.convert_to_tensor(np.ones(shape=[1, 1], dtype = np.int32))])
      self._discriminator.summary()

  def train(self, **kwargs):

    trainData = kwargs.pop("trainData")

    learningRate = kwargs.pop("learningRate", 1e-4)
    learningRate_f = kwargs.pop("learningRate_f", 1e-6)
    beta1 = kwargs.pop("beta1", 0.0)
    beta2 = kwargs.pop("beta2", 0.99)
    emaDecay = kwargs.pop("emaDecay", 0.999)

    iteration_S = kwargs.pop("iteration_S", 0)
    iteration_E = kwargs.pop("iteration_E", 100000)
    batchSize = kwargs.pop("batchSize", 6)

    ganType = kwargs.pop("ganType", "gan-gp")
    dsWeight = kwargs.pop("dsWeight", 1)
    dsIter = kwargs.pop("dsIter", 100000)
    adversarialWeight = kwargs.pop("adversarialWeight", 1)
    r1Weight = kwargs.pop("r1Weight", 1)
    styleWeight = kwargs.pop("styleWeight", 1)
    cycleWeight = kwargs.pop("cycleWeight", 1)

    modelPath = kwargs.pop("modelPath", "./model")
    loggingName = kwargs.pop('loggingName', f'{self.__class__.__name__}_train_{time.strftime("%Y%m%d-%H%M%S")}')
    loggingFolder = kwargs.pop('loggingFolder', './logs')
    loggingSnapshotInterval = kwargs.pop("loggingSnapshotInterval", (iteration_E - iteration_S) // 1)
    validationFolder = kwargs.pop('validationFolder', './validation')
    #checkpointList = kwargs.pop('checkpointList', [-1, iteration_E])

    # optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = beta1, beta_2 = beta2, epsilon = 1e-08)
    e_optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = beta1, beta_2 = beta2, epsilon = 1e-08)
    f_optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate_f, beta_1 = beta1, beta_2 = beta2, epsilon = 1e-08)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate, beta_1 = beta1, beta_2 = beta2, epsilon = 1e-08)

    logJson = {}
    if os.path.isdir(loggingFolder) == False:
      os.makedirs(loggingFolder)
    if os.path.isdir(validationFolder) == False:
      os.makedirs(validationFolder)
    logger = get_logger(loggingName, loggingFolder)

    # ckpt
    ckptdir = os.path.join(modelPath, "ckpt")
    os.makedirs(ckptdir, exist_ok = True)
    ckpt = tf.train.Checkpoint(generator = self._generator, generator_ema = self._generator_ema,
                              mappingNet = self._mappingNet, mappingNet_ema = self._mappingNet_ema,
                              styleEncoder = self._styleEncoder, styleEncoder_ema = self._styleEncoder_ema,
                              discriminator = self._discriminator,
                              g_optimizer = g_optimizer, e_optimizer = e_optimizer, 
                              f_optimizer = f_optimizer, d_optimizer = d_optimizer)
    ckptManager = tf.train.CheckpointManager(ckpt, ckptdir, max_to_keep = 1)

    if ckptManager.latest_checkpoint:
      ckpt.restore(ckptManager.latest_checkpoint).expect_partial()
      iteration_S = int(ckptManager.latest_checkpoint.split('-')[-1])

      logger.info(f"checkpoint restored from {ckptdir}")
      logger.info(f"start iteration was set to {iteration_S}")

    logJson['generator_params'] = self._generator.get_params()
    logger.info('{}'.format(json.dumps({'generator_params': logJson['generator_params']}, cls = JsonEncoder, indent = 2)[1 : -1].strip()))
    logJson['mappingNet_params'] = self._mappingNet.get_params()
    logger.info('{}'.format(json.dumps({'mappingNet_params': logJson['mappingNet_params']}, cls = JsonEncoder, indent = 2)[1 : -1].strip()))
    logJson['styleEncoder_params'] = self._styleEncoder.get_params()
    logger.info('{}'.format(json.dumps({'styleEncoder_params': logJson['styleEncoder_params']}, cls = JsonEncoder, indent = 2)[1 : -1].strip()))
    logJson['discriminator_params'] = self._discriminator.get_params()
    logger.info('{}'.format(json.dumps({'discriminator_params': logJson['discriminator_params']}, cls = JsonEncoder, indent = 2)[1 : -1].strip()))

    logJson['train_params'] = {
      'time': time.strftime('%Y%m%d %H:%M:%S'),
      'iteration_S': iteration_S,
      'iteration_E': iteration_E,
      'batchSize': batchSize,
      'emaDecay': emaDecay,

      'ganType': ganType,
      'dsWeight': dsWeight,
      'dsIter': dsIter,
      'adversarialWeight': adversarialWeight,
      'r1Weight': r1Weight,
      'styleWeight': styleWeight,
      'cycleWeight': cycleWeight,

      'g_optimizer': g_optimizer.get_config(),
      'e_optimizer': e_optimizer.get_config(),
      'f_optimizer': f_optimizer.get_config(),
      'd_optimizer': d_optimizer.get_config(),

      'loggingSnapshotInterval': loggingSnapshotInterval
    }
    logger.info('{}'.format(json.dumps({'train_params': logJson['train_params']}, cls = JsonEncoder, indent = 2)[1 : -1].strip()))

    logJson['trainData_params'] = trainData.get_log()
    logger.info('{}'.format(json.dumps({'trainData_params': logJson['trainData_params']}, cls = JsonEncoder, indent = 2)[1 : -1].strip()))

    logJsonSnapshots = []
    logger.info('Start optimization')

    """if -1 in checkpointList:
      self._generator.save_weights(os.path.join(ckptdir, "init_snapshot_generator"))
      self._generator_ema.save_weights(os.path.join(ckptdir, "init_snapshot_generator_ema"))
      self._mappingNet.save_weights(os.path.join(ckptdir, "init_snapshot_mappingNet"))
      self._mappingNet_ema.save_weights(os.path.join(ckptdir, "init_snapshot_mappingNet_ema"))
      self._styleEncoder.save_weights(os.path.join(ckptdir, "init_snapshot_styleEncoder"))
      self._styleEncoder_ema.save_weights(os.path.join(ckptdir, "init_snapshot_styleEncoder_ema"))
      self._discriminator.save_weights(os.path.join(ckptdir, "init_snapshot_discriminator"))"""

    @tf.function
    def g_train_step(xRl, yOg, yTg, zTgs = None, xRfs = None):
      with tf.GradientTape(persistent = True) as gTape:

        if zTgs is not None:
          zTg0, zTg1 = zTgs
        if xRfs is not None:
          xRf0, xRf1 = xRfs

        # adversarial loss
        if zTgs is not None:
          sTg = self._mappingNet([zTg0, yTg])
        else:
          sTg = self._styleEncoder([xRf0, yTg])

        xFake = self._generator([xRl, sTg])
        fakeLogit = self._discriminator([xFake, yTg])
        g_adv_loss = adversarialWeight * generator_loss(ganType, fakeLogit)

        # style reconstruction loss
        sPred = self._styleEncoder([xFake, yTg])
        g_sty_loss = styleWeight * L1_loss(sPred, sTg)

        # diversity sensitive loss
        if zTgs is not None:
          sTg1 = self._mappingNet([zTg1, yTg])
        else:
          sTg1 = self._styleEncoder([xRf1, yTg])
        
        xFake1 = self._generator([xRl, sTg1])
        xFake1 = tf.stop_gradient(xFake1)
        g_ds_loss = -dsWeight * L1_loss(xFake, xFake1)

        # cycle-consistency loss
        sOg = self._styleEncoder([xRl, yOg])
        xRec = self._generator([xFake, sOg])
        g_cyc_loss = cycleWeight * L1_loss(xRec, xRl)

        regular_loss = regularization_loss(self._generator)

        g_loss = g_adv_loss + g_sty_loss + g_ds_loss + g_cyc_loss + regular_loss

      gGrads = gTape.gradient(g_loss, self._generator.trainable_variables)
      g_optimizer.apply_gradients(zip(gGrads, self._generator.trainable_variables))

      if zTgs is not None:
        fGrads = gTape.gradient(g_loss, self._mappingNet.trainable_variables)
        eGrads = gTape.gradient(g_loss, self._styleEncoder.trainable_variables)

        f_optimizer.apply_gradients(zip(fGrads, self._mappingNet.trainable_variables))
        e_optimizer.apply_gradients(zip(eGrads, self._styleEncoder.trainable_variables))
      
      return g_adv_loss, g_sty_loss, g_ds_loss, g_cyc_loss, g_loss

    @tf.function
    def d_train_step(xRl, yOg, yTg, zTg = None, xRf = None):
      with tf.GradientTape() as dTape:

        if zTg is not None:
          sTg = self._mappingNet([zTg, yTg])
        else:
          sTg = self._styleEncoder([xRf, yTg])

        xFake = self._generator([xRl, sTg])

        realLogit = self._discriminator([xRl, yOg])
        fakeLogit = self._discriminator([xFake, yTg])

        d_adv_loss = adversarialWeight * discriminator_loss(ganType, realLogit, fakeLogit)
        if ganType == 'gan-gp':
          d_adv_loss += r1Weight * r1_gp_req(self._discriminator, xRl, yOg)
        regular_loss = regularization_loss(self._discriminator)

        d_loss = d_adv_loss + regular_loss
      
      dGrads = dTape.gradient(d_loss, self._discriminator.trainable_variables)
      d_optimizer.apply_gradients(zip(dGrads, self._discriminator.trainable_variables))

      return d_adv_loss, d_loss

    trainDataTFdata = trainData.create_tf_data()
    #trainDataTFdata = iter(trainDataTFdata.shuffle(buffer_size = trainData.numData, reshuffle_each_iteration = True).repeat().batch(batchSize))
    trainDataTFdata = iter(trainDataTFdata.repeat().batch(batchSize))

    dsWeightInit = dsWeight

    for iteridx in tqdm.tqdm(range(iteration_S, iteration_E), total = iteration_E - iteration_S):

      if dsWeight > 0:
        dsWeight = dsWeightInit - (dsWeightInit / dsIter) * iteridx

      xReal, _, yOrg = next(trainDataTFdata)
      xRef, xRef2, yTrg = next(trainDataTFdata)

      latent_fake_save_path = os.path.join(validationFolder, f"latent_{iteridx:08d}.jpg")
      ref_fake_save_path = os.path.join(validationFolder, f"ref_{iteridx:08d}.jpg")

      zTrg = tf.random.normal(shape=[batchSize, self._latentDim])
      zTrg2 = tf.random.normal(shape=[batchSize, self._latentDim])

      # update discriminator
      d_adv_loss_latent, d_loss_latent = d_train_step(xReal, yOrg, yTrg, zTg = zTrg)
      d_adv_loss_ref, d_loss_ref = d_train_step(xReal, yOrg, yTrg, xRf = xRef)

      # update generator
      g_adv_loss_latent, g_sty_loss_latent, g_ds_loss_latent, g_cyc_loss_latent, g_loss_latent = g_train_step(xReal, yOrg, yTrg, zTgs=[zTrg, zTrg2])
      g_adv_loss_ref, g_sty_loss_ref, g_ds_loss_ref, g_cyc_loss_ref, g_loss_ref = g_train_step(xReal, yOrg, yTrg, xRfs=[xRef, xRef2])

      # compute moving average of network parameters
      moving_average(self._generator, self._generator_ema, beta = emaDecay)
      moving_average(self._mappingNet, self._mappingNet_ema, beta = emaDecay)
      moving_average(self._styleEncoder, self._styleEncoder_ema, beta = emaDecay)

      logJsonSnapshot = {
          'totalIter': iteration_E - iteration_S,
          'iter': iteridx,
          'g_adv_loss_latent': f"{g_adv_loss_latent.numpy():.6f}",
          'g_sty_loss_latent': f"{g_sty_loss_latent.numpy():.6f}",
          'g_ds_loss_latent': f"{g_ds_loss_latent.numpy():.6f}", 
          'g_cyc_loss_latent': f"{g_cyc_loss_latent.numpy():.6f}",
          'g_loss_latent': f"{g_loss_latent.numpy():.6f}", 

          'g_adv_loss_ref': f"{g_adv_loss_ref.numpy():.6f}", 
          'g_sty_loss_ref': f"{g_sty_loss_ref.numpy():.6f}",
          'g_ds_loss_ref': f"{g_ds_loss_ref.numpy():.6f}",
          'g_cyc_loss_ref': f"{g_cyc_loss_ref.numpy():.6f}", 
          'g_loss_ref': f"{g_loss_ref.numpy():.6f}",

          'dsWeight': f"{dsWeight:.6f}",

          'd_adv_loss_latent': f"{d_adv_loss_latent.numpy():.6f}",
          'd_loss_latent': f"{d_loss_latent.numpy():.6f}",

          'd_adv_loss_ref': f"{d_adv_loss_ref.numpy():.6f}",
          'd_loss_ref': f"{d_loss_ref.numpy():.6f}",
      }
      logJsonSnapshots.append(logJsonSnapshot)

      if iteridx == iteration_S:
        xRealVal, xRefVal, yTrgVal = tf.identity(xReal), tf.identity(xRef), tf.identity(yTrg)

      if iteridx > 0 and iteridx % loggingSnapshotInterval == 0:
        logger.info('{}'.format(json.dumps(logJsonSnapshot, cls = JsonEncoder, indent = 2)))

        ckptManager.save(checkpoint_number = iteridx + 1)

      if iteridx % 100 == 0:
        latent_fake_save_path = os.path.join(validationFolder, f"latent_{iteridx:08d}.jpg")
        ref_fake_save_path = os.path.join(validationFolder, f"ref_{iteridx:08d}.jpg")

        self.latent_canvas(xRealVal, latent_fake_save_path)
        self.refer_canvas(xRealVal, xRefVal, yTrgVal, ref_fake_save_path, img_num = 5, batchSize = batchSize)
      

    logJson['train'] = logJsonSnapshots
    with open(os.path.join(loggingFolder, f'{loggingName}.json'), 'wt') as fout:
      fout.write(json.dumps(logJson, cls = JsonEncoder, indent = 2))

  def latent_canvas(self, x_real, path, returnOut = False, domain_fix_list = None):
    if domain_fix_list is None:
      domain_fix_list = tf.constant([idx for idx in range(self._numDomains)])
      _numDomains = self._numDomains
    else:
      domain_fix_list = tf.constant(domain_fix_list)
      _numDomains = len(domain_fix_list)
      
    canvas = PIL.Image.new('RGB', (self._imgSize * (_numDomains + 1) + 10, self._imgSize * self._numStyle), 'white')

    x_real = tf.expand_dims(x_real[0], axis=0)
    src_image = postprocess_images(x_real)[0]
    canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), (0, 0))

    z_trgs = tf.random.normal(shape=[self._numStyle, self._latentDim])

    for row in range(self._numStyle):
      z_trg = tf.expand_dims(z_trgs[row], axis=0)

      for col, y_trg in enumerate(list(domain_fix_list)):
        y_trg = tf.reshape(y_trg, shape=[1, 1])
        s_trg = self._mappingNet_ema([z_trg, y_trg])
        x_fake = self._generator_ema([x_real, s_trg])
        x_fake = postprocess_images(x_fake)

        col_image = x_fake[0]

        canvas.paste(PIL.Image.fromarray(np.uint8(col_image), 'RGB'), ((col + 1) * self._imgSize + 10, row * self._imgSize))

    if not returnOut:
      canvas.save(path)
    else:
      return np.array(canvas)

  def refer_canvas(self, x_real, x_ref, y_trg, path, img_num, batchSize):
    if type(img_num) == list:
      # In test phase
      src_img_num = img_num[0]
      ref_img_num = img_num[1]
    else:
      src_img_num = min(img_num, batchSize)
      ref_img_num = min(img_num, batchSize)

    x_real = x_real[:src_img_num]
    x_ref = x_ref[:ref_img_num]
    y_trg = y_trg[:ref_img_num]

    canvas = PIL.Image.new('RGB', (self._imgSize * (src_img_num + 1) + 10, self._imgSize * (ref_img_num + 1) + 10),
                            'white')

    x_real_post = postprocess_images(x_real)
    x_ref_post = postprocess_images(x_ref)

    for col, src_image in enumerate(list(x_real_post)):
      canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * self._imgSize + 10, 0))

    for row, dst_image in enumerate(list(x_ref_post)):
      canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * self._imgSize + 10))

      row_images = np.stack([dst_image] * src_img_num)
      row_images = preprocess_fit_train_image(row_images)
      row_images_y = np.stack([y_trg[row]] * src_img_num)

      s_trg = self._styleEncoder_ema([row_images, row_images_y])
      row_fake_images = postprocess_images(self._generator_ema([x_real, s_trg]))

      for col, image in enumerate(list(row_fake_images)):
        canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'),
                      ((col + 1) * self._imgSize + 10, (row + 1) * self._imgSize + 10))

    canvas.save(path)