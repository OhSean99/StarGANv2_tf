from turtle import down
import tensorflow as tf

import numpy as np

from networks.basic_layers import FC_L, Conv_L, InstanceNorm, Residual_Block, AdaIN_Residual_Block

class DISCRIMINATOR(tf.keras.Model):
  def __init__(self, **kwargs):
    super(DISCRIMINATOR, self).__init__(name = "DISCRIMINATOR")

    self._imgSize = kwargs.pop("imgSize", 256)
    self._numDomains = kwargs.pop("numDomains", 2)
    self._maxConvDim = kwargs.pop("maxConvDim", 512)
    self._sn = kwargs.pop("sn", False)

    self._channels = 2 ** 14 // self._imgSize
    self._repeatNum = int(np.log2(self._imgSize)) - 2

    inCh = outCh = self._channels

    templayers = [Conv_L(numFeats = inCh,
                        kernelSize = 3,
                        strides = 1,
                        padding = 1,
                        sn = self._sn,
                        name = "init_conv")]

    for ii in range(self._repeatNum):
      outCh = min(inCh * 2, self._maxConvDim)
      templayers.append(Residual_Block(inCh = inCh,
                                      outCh = outCh,
                                      downsample = True,
                                      sn = self._sn,
                                      name = f"resblock_{ii}"))
      inCh = outCh
    
    templayers += [tf.keras.layers.LeakyReLU(alpha = 0.2),
                  Conv_L(numFeats = outCh,
                        kernelSize = 4,
                        strides = 1,
                        padding = 0,
                        sn = self._sn,
                        name = "conv_0"),
                  tf.keras.layers.LeakyReLU(alpha = 0.2),
                  Conv_L(numFeats = self._numDomains,
                        kernelSize = 1,
                        strides = 1,
                        sn = self._sn,
                        name = "conv_1")]

    self._encoder = tf.keras.Sequential(templayers)
    del templayers

  def call(self, xIn):

    x, domain = xIn

    x = self._encoder(x)
    x = tf.reshape(x, shape = [x.shape[0], -1])

    x = tf.gather(x, domain, axis = 1, batch_dims = -1)

    return x

  def get_params(self):

    return {"imgSize": self._imgSize,
            "numDomains": self._numDomains,
            "maxConvDim": self._maxConvDim,
            "spectralNormalization": self._sn}

class STYLEENCODER(tf.keras.Model):
  def __init__(self, **kwargs):
    super(STYLEENCODER, self).__init__(name = "STYLEENCODER")

    self._imgSize = kwargs.pop("imgSize", 256)
    self._styleDim = kwargs.pop("styleDim", 64)
    self._numDomains = kwargs.pop("numDomains", 2)
    self._maxConvDim = kwargs.pop("maxConvDim", 512)
    self._sn = kwargs.pop("sn", False)

    self._channels = 2 ** 14 // self._imgSize
    self._repeatNum = int(np.log2(self._imgSize)) - 2

    inCh = outCh = self._channels
    
    templayers = [Conv_L(numFeats = inCh,
                        kernelSize = 3,
                        strides = 1,
                        padding = 1,
                        sn = self._sn,
                        name = "init_conv")]

    for ii in range(self._repeatNum):
      outCh = min(inCh * 2, self._maxConvDim)
      templayers.append(Residual_Block(inCh = inCh,
                                      outCh = outCh,
                                      downsample = True,
                                      sn = self._sn,
                                      name = f"resblock_{ii}"))
      inCh = outCh

    templayers += [tf.keras.layers.LeakyReLU(alpha = 0.2),
                  Conv_L(numFeats = outCh,
                        kernelSize = 4,
                        strides = 1,
                        padding = 0,
                        sn = self._sn,
                        name = "conv"),
                  tf.keras.layers.LeakyReLU(alpha = 0.2)]
    
    self._sharedLayers = tf.keras.Sequential(templayers)
    del templayers
    self._unsharedLayers = [FC_L(nunits = self._styleDim, sn = self._sn, name = f"domain_{nd}_styleFC") for nd in range(self._numDomains)]

  def call(self, xIn):
    x, domain = xIn

    h = self._sharedLayers(x)

    x = []
    for l0 in self._unsharedLayers:
      x += [l0(h)]

    x = tf.stack(x, axis = 1)
    x = tf.gather(x, domain, axis = 1, batch_dims = -1)
    x = tf.squeeze(x, axis = 1)

    return x

  def get_params(self):

    return {"imgSize": self._imgSize,
            "styleDim": self._styleDim,
            "numDomains": self._numDomains,
            "maxConvDim": self._maxConvDim,
            "spectralNormalization": self._sn}

class MAPPINGNETWORK(tf.keras.Model):
  def __init__(self, **kwargs):
    super(MAPPINGNETWORK, self).__init__(name = "MAPPINGNETWORK")

    self._styleDim = kwargs.pop("styleDim", 64)
    self._hiddenDim = kwargs.pop("hiddenDim", 512)
    self._numDomains = kwargs.pop("numDomains", 2)
    self._sn = kwargs.pop("sn", False)

    self._sharedLayers, self._unsharedLayers = [], []

    templayers = []
    for ii in range(4):
      templayers += self.FC_relu(f"sharedFC_{ii}")
    self._sharedLayers = tf.keras.Sequential(templayers)

    templayers = []
    for nd in range(self._numDomains):
      for ii in range(3):
        templayers += self.FC_relu(f"domain_{nd}_unsharedFC_{ii}")
      templayers.append(FC_L(nunits = self._styleDim, sn = self._sn, name = f"domain_{nd}_styleFC"))
      self._unsharedLayers += [tf.keras.Sequential(templayers)]
    del templayers

  def FC_relu(self, name):
    return [FC_L(nunits = self._hiddenDim, sn = self._sn, name = name),
            tf.keras.layers.ReLU()]

  def call(self, xIn):
    z, domain = xIn

    h = self._sharedLayers(z)
    
    x = []
    for l0 in self._unsharedLayers:
      x += [l0(h)]

    x = tf.stack(x, axis = 1)
    x = tf.gather(x, domain, axis = 1, batch_dims = -1)
    x = tf.squeeze(x, axis = 1)

    return x
  
  def get_params(self):

    return {"styleDim": self._styleDim,
            "hiddenDim": self._hiddenDim,
            "numDomains": self._numDomains,
            "spectralNormalization": self._sn}

class GENERATOR(tf.keras.Model):
  def __init__(self, **kwargs):
    super(GENERATOR, self).__init__(name = "GENRATOR")

    self._imgSize = kwargs.pop("imgSize", 256)
    self._imgCh = kwargs.pop("imgCh", 3)
    self._styleDim = kwargs.pop("styleDim", 64)
    self._maxConvDim = kwargs.pop("maxConvDim", 512)
    self._sn = kwargs.pop("sn", False)

    self._channels = 2 ** 14 // self._imgSize
    self._repeatNum = int(np.log2(self._imgSize)) - 4

    self._RGBInConv = Conv_L(numFeats = self._channels,
                            kernelSize = 3,
                            strides = 1,
                            padding = 1,
                            sn = self._sn,
                            name = "RGB_in_conv")

    self._encoder, self._decoder = [], []
    inCh = outCh = self._channels
    for ii in range(self._repeatNum):
      outCh = min(inCh * 2, self._maxConvDim)

      self._encoder.append(Residual_Block(inCh = inCh, outCh = outCh,
                                          normalize = True,
                                          downsample = True,
                                          sn = self._sn,
                                          name = f"encoder_down_resblock_{ii}"))
      #self._decoder.insert(0, AdaIN_Residual_Block(inCh = outCh, outCh = inCh,
      #                                            upsample = True,
      #                                            sn = self._sn,
      #                                            name = f"decoder_up_adaINresblock_{ii}"))
      self._decoder.append(AdaIN_Residual_Block(inCh = outCh, outCh = inCh,
                                                upsample = True,
                                                sn = self._sn,
                                                name = f"decoder_up_adaINresblock_{ii}"))
      inCh = outCh
    
    for ii in range(2):
      self._encoder.append(Residual_Block(inCh = outCh, outCh = outCh,
                                          normalize = True,
                                          sn = self._sn,
                                          name = f"encoder_bottleneck_resblock_{ii}"))
      #self._decoder.insert(0, AdaIN_Residual_Block(inCh = outCh, outCh = outCh,
      #                                            sn = self._sn,
      #                                            name = f"decoder_bottleneck_adaINresblock_{ii}"))
      self._decoder.append(AdaIN_Residual_Block(inCh = outCh, outCh = outCh,
                                                  sn = self._sn,
                                                  name = f"decoder_bottleneck_adaINresblock_{ii}"))

    self._RGBOutConv = tf.keras.Sequential([InstanceNorm(),
                                            tf.keras.layers.LeakyReLU(alpha = 0.2),
                                            Conv_L(numFeats = self._imgCh,
                                                  kernelSize = 1,
                                                  strides = 1,
                                                  sn = self._sn)],
                                          name = "RGB_out_conv")


  def call(self, xIn):
    
    x, xs = xIn

    x = self._RGBInConv(x)

    for encblk in self._encoder:
      x = encblk(x)
    
    for decidx in range(len(self._decoder) - 1, -1, -1):
      x = self._decoder[decidx]([x, xs])
    #for decblk in self._decoder:
    #  x = decblk([x, xs])

    x = self._RGBOutConv(x)

    return x    

  def get_params(self):

    return {"imgSizse": self._imgSize,
            "imgCh": self._imgCh,
            "styleDim": self._styleDim,
            "maxConvDim": self._maxConvDim,
            "spectralNormalization": self._sn}