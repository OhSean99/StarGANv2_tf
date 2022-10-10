import tensorflow as tf

def r1_gp_req(discriminator, x_real, y_org):
  with tf.GradientTape() as p_tape:
    p_tape.watch(x_real)
    real_loss = tf.reduce_sum(discriminator([x_real, y_org]))

  real_grads = p_tape.gradient(real_loss, x_real)

  r1_penalty = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))

  return r1_penalty

def regularization_loss(model):
  loss = tf.nn.scale_regularization_loss(model.losses)

  return loss

def discriminator_loss(gan_type, real_logit, fake_logit):

  real_loss = 0
  fake_loss = 0

  if gan_type == 'lsgan' :
    real_loss = tf.reduce_mean(tf.math.squared_difference(real_logit, 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake_logit))

  if gan_type == 'gan' or gan_type == 'gan-gp' :
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))

  if gan_type == 'hinge' :
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))

  return real_loss + fake_loss

def generator_loss(gan_type, fake_logit):
  fake_loss = 0

  if gan_type == 'lsgan' :
    fake_loss = tf.reduce_mean(tf.math.squared_difference(fake_logit, 1.0))

  if gan_type == 'gan' or gan_type == 'gan-gp':
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))

  if gan_type == 'hinge' :
    fake_loss = -tf.reduce_mean(fake_logit)

  return fake_loss

def L1_loss(x, y):
  loss = tf.reduce_mean(tf.abs(x - y))

  return loss

@tf.function
def moving_average(model, model_test, beta=0.999):
  for param, param_test in zip(model.trainable_weights, model_test.trainable_weights):
    param_test.assign(lerp(param, param_test, beta))

def lerp(a, b, t):
  out = a + (b - a) * t
  return out