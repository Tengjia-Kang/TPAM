import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import tensorflow_addons as tfa
from utils import utils
from easydict import EasyDict
layers = tf.keras.layers
import numpy as np
class RnnWalkBase(tf.keras.Model):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn=None,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnWalkBase, self).__init__(name='')

    self._classes = classes
    self._params = params
    self._model_must_be_load = model_must_be_load

    # self._pooling_betwin_grus = 'pooling' in self._params.aditional_network_params
    # self._bidirectional_rnn = 'bidirectional_rnn' in self._params.aditional_network_params

    self._init_layers()
    inputs = tf.keras.layers.Input(shape=(100, net_input_dim))
    self.build(input_shape=(1, 100, net_input_dim))
    outputs = self.call(inputs)
    if dump_model_visualization:
      tmp_model = keras.Model(inputs=inputs, outputs=outputs, name='WalkModel')
      tmp_model.summary(print_fn=self._print_fn)
      tf.keras.utils.plot_model(tmp_model, params.logdir + '/RnnWalkModel.png', show_shapes=True)

    self.manager = None
    if optimizer:
      if model_fn:
        #self.checkpoint = tf.train.Checkpoint(optimizer=copy.deepcopy(optimizer), model=self)
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      else:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self._params.logdir, max_to_keep=5)
      if model_fn: # Transfer learning
        self.load_weights(model_fn)
        self.checkpoint.optimizer = optimizer
      else:
        self.load_weights()
    else:
      self.checkpoint = tf.train.Checkpoint(model=self)
      if model_fn:
        self.load_weights(model_fn)
      else:
        self.load_weights(tf.train.latest_checkpoint(self._params.logdir))

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def load_weights(self, filepath=None):
    if filepath is not None and filepath.endswith('.keras'):
      super(RnnWalkBase, self).load_weights(filepath)
    elif filepath is None:
      status = self.checkpoint.restore(self.manager.latest_checkpoint)
      print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(), utils.color.END)
    else:
      filepath = filepath.replace('//', '/')
      status = self.checkpoint.restore(filepath)

  def save_weights(self, folder, step=None, keep=False):
    if self.manager is not None:
      self.manager.save()
    if keep:
      super(RnnWalkBase, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')
      #self.checkpoint.write(folder + '/learned_model2keep--' + str(step))

class RnnMixtureNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnMixtureNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = tf.keras.layers.BatchNormalization(axis=2)
      self._norm2 = tf.keras.layers.BatchNormalization(axis=2)
    # self._fc1 = tf.keras.layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
    #                          kernel_initializer=initializer)
    self._fc1 = tfp.layers.DenseReparameterization(self._layer_sizes['fc1'])
    self._fc2 = tfp.layers.DenseReparameterization(self._layer_sizes['fc2'])
    # rnn_layer = layers.LSTM
    rnn_layer = tf.keras.layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru3 = layers.Bidirectional(self._gru3)
    #   print('Using Bidirectional GRUs.')

    # self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
    #                              kernel_initializer=initializer)
    self._fc_last = tf.keras.layers.Dense(self._classes, activation= "softmax",
                                 kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._norm_input = False
    if self._norm_input:
      self._norm_features = tf.keras.layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    # if self._pooling_betwin_grus:
    #   x1 = self._pooling(x1)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    # if self._pooling_betwin_grus:
    #   x2 = self._pooling(x2)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    #if self._params.one_label_per_model:
    #  x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

class RnnWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = layers.BatchNormalization(axis=2)
      self._norm2 = layers.BatchNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)

    self._fc1_prob = tfp.layers.DenseReparameterization(
      layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                   kernel_initializer=initializer))


    self._fc2_prob =tfp.layers.DenseReparameterization(layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer))
    # rnn_layer = layers.LSTM
    rnn_layer = layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru3 = layers.Bidirectional(self._gru3)
    #   print('Using Bidirectional GRUs.')

    # self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
    #                              kernel_initializer=initializer)
    self._fc_last = layers.Dense(self._classes, activation= "softmax",
                                 kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._norm_input = False
    if self._norm_input:
      self._norm_features = layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    # if self._pooling_betwin_grus:
    #   x1 = self._pooling(x1)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    # if self._pooling_betwin_grus:
    #   x2 = self._pooling(x2)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    #if self._params.one_label_per_model:
    #  x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x
class RnnMixtureNet_First_Bayes(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnMixtureNet_First_Bayes, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = tf.keras.layers.BatchNormalization(axis=2)
      self._norm2 = tf.keras.layers.BatchNormalization(axis=2)
    self._fc1 = tf.keras.layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    # self._fc1 = tfp.layers.DenseReparameterization(self._layer_sizes['fc1'])
    self._fc2 = tfp.layers.DenseReparameterization(self._layer_sizes['fc2'])
    # rnn_layer = layers.LSTM
    rnn_layer = tf.keras.layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru3 = layers.Bidirectional(self._gru3)
    #   print('Using Bidirectional GRUs.')

    # self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
    #                              kernel_initializer=initializer)
    self._fc_last = tf.keras.layers.Dense(self._classes, activation= "softmax",
                                 kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._norm_input = False
    if self._norm_input:
      self._norm_features = tf.keras.layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    # if self._pooling_betwin_grus:
    #   x1 = self._pooling(x1)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    # if self._pooling_betwin_grus:
    #   x2 = self._pooling(x2)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    #if self._params.one_label_per_model:
    #  x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x
class RnnMixtureNet_Second_Bayes(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnMixtureNet_Second_Bayes, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = tf.keras.layers.BatchNormalization(axis=2)
      self._norm2 = tf.keras.layers.BatchNormalization(axis=2)
    # self._fc1 = tf.keras.layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
    #                          kernel_initializer=initializer)
    self._fc1 = tfp.layers.DenseReparameterization(self._layer_sizes['fc1'])
    # self._fc2 = tfp.layers.DenseReparameterization(self._layer_sizes['fc2'])
    self._fc2 = tf.keras.layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    # rnn_layer = layers.LSTM
    rnn_layer = tf.keras.layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru1 = layers.Bidirectional(self._gru1)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            #trainable=False,
                            #activation='sigmoid',
                            dropout=self._params.net_gru_dropout,
                            #recurrent_dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru2 = layers.Bidirectional(self._gru2)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           #trainable=False,
                           #activation='sigmoid',
                           dropout=self._params.net_gru_dropout,
                           #recurrent_dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    # if self._bidirectional_rnn:
    #   self._gru3 = layers.Bidirectional(self._gru3)
    #   print('Using Bidirectional GRUs.')

    # self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
    #                              kernel_initializer=initializer)
    self._fc_last = tf.keras.layers.Dense(self._classes, activation= "softmax",
                                 kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    self._norm_input = False
    if self._norm_input:
      self._norm_features = tf.keras.layers.LayerNormalization(axis=-1, trainable=False)

  # @tf.function
  def call(self, model_ftrs, classify=True, skip_1st=True, training=True, mask=None):
    if self._norm_input:
      model_ftrs = self._norm_features(model_ftrs)
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    x1 = self._gru1(x, training=training)
    # if self._pooling_betwin_grus:
    #   x1 = self._pooling(x1)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x2 = self._gru2(x1, training=training)
    # if self._pooling_betwin_grus:
    #   x2 = self._pooling(x2)
    #   if mask is not None:
    #     mask = mask[:, ::2]
    x3 = self._gru3(x2, training=training, mask=mask)
    x = x3

    #if self._params.one_label_per_model:
    #  x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x

  def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
    if skip_1st:
      x = model_ftrs[:, 1:]
    else:
      x = model_ftrs
    if get_layer == 'input':
      return x
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc1':
      return x
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)
    if get_layer == 'fc2':
      return x
    x = self._gru1(x, training=training)
    if get_layer == 'gru1':
      return x
    x = self._gru2(x, training=training)
    if get_layer == 'gru2':
      return x
    x = self._gru3(x, training=training)
    if get_layer == 'gru3':
      return x

    if self._params.one_label_per_model:
      x = x[:, -1, :]

    if classify:
      x = self._fc_last(x)
    return x


def show_model():
  def fn(to_print):
    print(to_print)
  if 1:
    params = EasyDict({'n_classes': 3, 'net_input_dim': 3, 'batch_size': 32, 'last_layer_activation': 'softmax',
                       'one_label_per_model': True, 'logdir': '.'})
    params.net_input_dim = 3 + 5
    model = RnnWalkNet(params, classes=3, net_input_dim=3, model_fn=None)
  else:
    model = set_up_rnn_walk_model()
    tf.keras.utils.plot_model(model, "RnnWalkModel.png", show_shapes=True)
    model.summary(print_fn=fn)



if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu(0)
  show_model()