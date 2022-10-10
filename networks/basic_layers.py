import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import math

def pytorch_kaiming_weight_factor(a = 0.0, activation_function = 'relu') :

  if activation_function == 'relu' :
    gain = np.sqrt(2.0)
  elif activation_function == 'leaky_relu' :
    gain = np.sqrt(2.0 / (1 + a ** 2))
  elif activation_function =='tanh' :
    gain = 5.0 / 3
  else :
    gain = 1.0

  factor = gain * gain
  mode = 'fan_in'

  return factor, mode

_factor, _mode = pytorch_kaiming_weight_factor(activation_function = 'relu')
_distribution = "untruncated_normal"
# distribution in {"uniform", "truncated_normal", "untruncated_normal"}
weightInitializer = tf.initializers.VarianceScaling(scale = _factor, mode = _mode, distribution = _distribution)
weightRegularizer = tf.keras.regularizers.l2(1e-4)
weightRegularizerFC = tf.keras.regularizers.l2(1e-4)

def InstanceNorm(epsilon = 1e-5, name='InstanceNorm'):
  return tfa.layers.normalizations.InstanceNormalization(epsilon = epsilon, scale = True, center = True,
                                                          name = name)

def nearest_up_sample(x, scale_factor=2):
  _, h, w, _ = x.get_shape().as_list()
  new_size = [h * scale_factor, w * scale_factor]
  return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

class AdaIN(tf.keras.layers.Layer):
  def __init__(self, numFeats, sn = False, epsilon = 1e-5, name = "AdaIN"):
    super(AdaIN, self).__init__(name = name)
    self._numFeats = numFeats
    self._epsilon = epsilon

    self._gammaFC = FC_L(nunits = self._numFeats, addBias = True, sn = sn)
    self._betaFC = FC_L(nunits = self._numFeats, addBias = True, sn = sn)

  def call(self, xIn):
    x, style = xIn
    x_mean, x_var = tf.nn.moments(x, axes = [1, 2], keepdims = True)
    x_std = tf.sqrt(x_var + self._epsilon)

    x_norm = ((x - x_mean) / x_std)

    gamma = self._gammaFC(style)
    beta = self._betaFC(style)

    gamma = tf.reshape(gamma, shape = [-1, 1, 1, self._numFeats])
    beta = tf.reshape(beta, shape = [-1, 1, 1, self._numFeats])

    x = (1 + gamma) * x_norm + beta

    return x

class AdaIN_Residual_Block(tf.keras.layers.Layer):
  def __init__(self, inCh, outCh, 
                upsample = False, 
                addBias = True, sn = False, 
                name = "AdaIN_Residual_Block"):
    super(AdaIN_Residual_Block, self).__init__(name = name)

    self._upsample = upsample
    self._skip = inCh != outCh

    self._convl0 = Conv_L(numFeats = outCh,
                          kernelSize = 3,
                          strides = 1,
                          padding = 1,
                          addBias = addBias,
                          sn = sn,
                          name = "conv_0")
    self._adainl0 = AdaIN(inCh, name = "adain_0")
    self._convl1 = Conv_L(numFeats = outCh,
                          kernelSize = 3,
                          strides = 1,
                          padding = 1,
                          addBias = addBias,
                          sn = sn,
                          name = "conv_1")
    self._adainl1 = AdaIN(outCh, name = "adain_1")

    if self._skip:
      self._skipConvl = Conv_L(numFeats = outCh,
                              kernelSize = 1,
                              strides = 1,
                              addBias = False,
                              sn = sn,
                              name = "skip_conv")

  def shortcut(self, x):

    if self._upsample:
      x = nearest_up_sample(x, scale_factor = 2)
    if self._skip:
      x = self._skipConvl(x)
    
    return x
  
  def residual(self, x, s):
    
    x = self._adainl0([x, s])
    x = tf.nn.leaky_relu(x, alpha = 0.2)
    if self._upsample:
      x = nearest_up_sample(x, scale_factor = 2)
    x = self._convl0(x)

    x = self._adainl1([x, s])
    x = tf.nn.leaky_relu(x, alpha = 0.2)
    x = self._convl1(x)

    return x
  
  def call(self, xIn):
    x_c, x_s = xIn

    x = self.residual(x_c, x_s) + self.shortcut(x_c)

    return x / math.sqrt(2)


class Residual_Block(tf.keras.layers.Layer):
  def __init__(self, inCh, outCh, 
                normalize = False, downsample = False, 
                addBias = True, sn = False, name = "Residual_Block"):

    super(Residual_Block, self).__init__(name = name)
    self._normalize = normalize
    self._downsample = downsample
    self._skip = inCh != outCh

    self._convl0 = Conv_L(numFeats = inCh,
                          kernelSize = 3,
                          strides = 1,
                          padding = 1,
                          addBias = addBias,
                          sn = sn,
                          name = "conv_0")
    self._instNorml0 = InstanceNorm()

    self._convl1 = Conv_L(numFeats = outCh,
                          kernelSize = 3,
                          strides = 1,
                          padding = 1,
                          addBias = addBias,
                          sn = sn,
                          name = "conv_1")
    self._instNorml1 = InstanceNorm()

    if self._skip:
      self._skipConvl = Conv_L(numFeats = outCh,
                              kernelSize = 1,
                              strides = 1,
                              addBias = False,
                              sn = sn,
                              name = "skip_conv")

  def shortcut(self, x):
    if self._skip:
      x = self._skipConvl(x)
    if self._downsample:
      x = tf.nn.avg_pool2d(x, 2, 2, "VALID")
    
    return x
  
  def residual(self, x):
    if self._normalize:
      x = self._instNorml0(x)
    x = tf.nn.leaky_relu(x, alpha = 0.2)
    x = self._convl0(x)

    if self._downsample:
      x = tf.nn.avg_pool2d(x, 2, 2, "VALID")
    if self._normalize:
      x = self._instNorml1(x)
    x = tf.nn.leaky_relu(x, alpha = 0.2)
    x = self._convl1(x)

    return x
  
  def call(self, xIn):

    x = self.residual(xIn) + self.shortcut(xIn)

    return x / math.sqrt(2)

class FC_L(tf.keras.layers.Layer):
  def __init__(self, nunits, addBias = True, sn = False, name = "FC_L"):
    super(FC_L, self).__init__(name = name)

    if sn:
      self._fcl = Spectral_Normalization(tf.keras.layers.Dense(nunits,
                                                              kernel_initializer = weightInitializer,
                                                              kernel_regularizer = weightRegularizerFC,
                                                              use_bias = addBias),
                                        name = f"sn_{name}")
    else:
      self._fcl = tf.keras.layers.Dense(nunits,
                                        kernel_initializer = weightInitializer,
                                        kernel_regularizer = weightRegularizerFC,
                                        use_bias = addBias,
                                        name = f"sn_{name}")

    self._flatter = tf.keras.layers.Flatten()

  def call(self, x):
    x = self._flatter(x)
    return self._fcl(x)

class Conv_L(tf.keras.layers.Layer):
  def __init__(self, numFeats, kernelSize = 3, strides = 1, 
              padding = 0, padType = "zero", addBias = True, sn = False,
              name = "Conv_L"):
    super(Conv_L, self).__init__(name = name)
    self._padding = padding
    self._strides = strides
    self._kernelSize = kernelSize
    self._padType = padType

    if sn:
      self._convl = Spectral_Normalization(tf.keras.layers.Conv2D(filters = numFeats,
                                                                  kernel_size = kernelSize,
                                                                  kernel_initializer = weightInitializer,
                                                                  kernel_regularizer = weightRegularizer,
                                                                  strides = strides,
                                                                  use_bias = addBias),
                                            name = f'sn_{name}')
    else:
      self._convl = tf.keras.layers.Conv2D(filters = numFeats,
                                            kernel_size = kernelSize,
                                            kernel_initializer = weightInitializer,
                                            kernel_regularizer = weightRegularizer,
                                            strides = strides,
                                            use_bias = addBias,
                                            name = name)
  def call(self, x, training = None, mask = None):
    
    if self._padding > 0:
      h = x.shape[1]
      if h % self._strides == 0:
        pad = self._padding * 2
      else:
        pad = max(self._kernelSize - (h % self._strides), 0)

      padT = pad // 2
      padB = pad - padT
      padL = pad // 2
      padR = pad - padL

      if self._padType == 'reflect':
        x = tf.pad(x, [[0, 0], [padT, padB], [padL, padR], [0, 0]], mode='REFLECT')
      else:
        x = tf.pad(x, [[0, 0], [padT, padB], [padL, padR], [0, 0]])
    
    return self._convl(x)


class Spectral_Normalization(tf.keras.layers.Wrapper):

  def __init__(self, layer, iteration = 1, eps = 1e-12, training = True, **kwargs):
    self._iteration = iteration
    self._eps = eps
    self.do_power_iteration = training

    super(Spectral_Normalization, self).__init__(layer, **kwargs)
  
  def build(self, inputShape = None):
    self.layer.build(inputShape)

    self._w = self.layer.kernel
    self._wShape = self._w.shape.as_list()

    self._u = self.add_weight(shape=(1, self._wShape[-1]),
                              initializer = tf.initializers.TruncatedNormal(stddev = 0.02),
                              trainable = False,
                              name = self.name + '_u',
                              dtype = tf.float32, aggregation = tf.VariableAggregation.ONLY_FIRST_REPLICA)

    super(Spectral_Normalization, self).build()

  def call(self, inputs, training = None, mask = None):
    self.update_weights()
    output = self.layer(inputs)
    return output
  
  def update_weights(self):
    w_reshaped = tf.reshape(self._w, [-1, self._wShape[-1]])

    u_hat = self._u
    v_hat = None

    if self.do_power_iteration:
      for _ in range(self._iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
        v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self._eps)

        u_ = tf.matmul(v_hat, w_reshaped)
        u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self._eps)

    sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
    self._u.assign(u_hat)

    self.layer.kernel.assign(self._w / sigma)