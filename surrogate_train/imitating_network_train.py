import os
import time
import numpy
import tensorflow as tf
import tensorflow_addons as tfa

from models import beyesican_rnn_model, rnn_model
from data import dataset
from utils import utils
from surrogate_train import params_setting
import argparse

def label_to_one_hot(labels: tf.Tensor, params):
  return tf.one_hot(indices=labels, depth=params.n_classes)

def print_enters(to_print):
  print("\n\n\n\n")
  print(to_print)
  print("\n\n\n\n")

def train_val(params):
  build_info = tf.sysconfig.get_build_info()
  print(build_info)
  utils.next_iter_to_keep = 1000
  print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
  print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
  utils.backup_python_files_and_params(params)

  # Set up datasets_processed for training and for test
  # -----------------------------------------
  train_datasets = []
  train_ds_iters = []
  max_train_size = 0
  for i in range(len(params.datasets2use['train'])):
    if config['train_use_pred_vector'] is True:
      this_train_dataset, n_trn_items = dataset.tf_mesh_dataset_with_pred_vector(params, params.datasets2use['train'][i],
                                                                                 mode=params.network_tasks[i],
                                                                                 size_limit=params.train_dataset_size_limit,
                                                                                 shuffle_size=100,
                                                                                 min_max_faces2use=params.train_min_max_faces2use,
                                                                                 max_size_per_class=params.train_max_size_per_class,
                                                                                 min_dataset_size=128,
                                                                                 data_augmentation=params.train_data_augmentation)

    else:
      this_train_dataset, n_trn_items = dataset.tf_mesh_dataset(params, params.datasets2use['train'][i],
                                                                mode=params.network_tasks[i],
                                                                size_limit=params.train_dataset_size_limit,
                                                                shuffle_size=100,
                                                                min_max_faces2use=params.train_min_max_faces2use,
                                                                max_size_per_class=params.train_max_size_per_class,
                                                                min_dataset_size=128,
                                                                data_augmentation=params.train_data_augmentation)
    print('Train Dataset size:', n_trn_items)
    train_ds_iters.append(iter(this_train_dataset.repeat()))
    train_datasets.append(this_train_dataset)

    max_train_size = max(max_train_size, n_trn_items)
  train_epoch_size = max(16, int(max_train_size / params.n_walks_per_model / params.batch_size))
  print('train_epoch_size:', train_epoch_size)
  if params.datasets2use['test'] is None:
    test_dataset = None
    n_tst_items = 0
  else:
    test_dataset, n_tst_items = dataset.tf_mesh_dataset(params, params.datasets2use['test'][0],
                                                        mode=params.network_tasks[0],
                                                        size_limit=params.test_dataset_size_limit,
                                                        shuffle_size=100,
                                                        min_max_faces2use=params.test_min_max_faces2use)
    test_ds_iter = iter(test_dataset.repeat())
  print(' Test Dataset size:', n_tst_items)


  # Set up RNN model and optimizer
  if params.net_start_from_prev_net is not None:
    init_net_using = params.net_start_from_prev_net
  else:
    init_net_using = None

  if params.optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate, clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'cycle':
    @tf.function
    #learning rate调整策略，当x大于x_th时，lr将以0.5倍逐渐减小
    def _scale_fn(x):
      x_th = 500e3 / params.cycle_opt_prms.step_size
      if x < x_th:
        return 1.0
      else:
        return 0.5
    lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
                                                      maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
                                                      step_size=params.cycle_opt_prms.step_size,
                                                      scale_fn=_scale_fn, scale_mode="cycle", name="MyCyclicScheduler")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'sgd':
    optimizer = tf.keras.optimizers.SGD(lr=params.learning_rate[0], decay=0, momentum=0.9, nesterov=True,
                                        clipnorm=params.gradient_clip_th)
  else:
    raise Exception('optimizer_type not supported: ' + params.optimizer_type)

  if params.net == 'RnnWalkNet':
    if config['beyesican_dnn_training']:
      dnn_model = beyesican_rnn_model.RnnMixtureNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                                    optimizer=optimizer)
    else:
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                       optimizer=optimizer)
  elif params.net == 'RnnMixtureNet':
    dnn_model = beyesican_rnn_model.RnnMixtureNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                                  optimizer=optimizer)
  elif params.net == "Manifold_RnnWalkNet":
    if config['beyesican_dnn_training']:
      dnn_model = beyesican_rnn_model.RnnMixtureNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                                    optimizer=optimizer)
    else:
      dnn_model = rnn_model.RnnManifoldWalkNet(params, params.n_classes, params.net_input_dim, init_net_using,
                                               optimizer=optimizer)
  elif params.net == "RnnMixtureNet_first_bayes":
    if config['beyesican_dnn_training']:
      dnn_model = beyesican_rnn_model.RnnMixtureNet_First_Bayes(params, params.n_classes, params.net_input_dim, init_net_using,
                                                                optimizer=optimizer)
  elif params.net == "RnnMixtureNet_second_bayes":
    if config['beyesican_dnn_training']:
      dnn_model = beyesican_rnn_model.RnnMixtureNet_Second_Bayes(params, params.n_classes, params.net_input_dim, init_net_using,
                                                                 optimizer=optimizer)
  # Other initializations
  # ---------------------
  time_msrs = {}
  time_msrs_names = ['train_step', 'get_train_data', 'test']
  for name in time_msrs_names:
    time_msrs[name] = 0
  manifold_seg_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='seg_train_accuracy')
  seg_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='seg_train_accuracy')

  train_log_names = ['seg_loss']
  train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
  train_logs['seg_train_accuracy'] = seg_train_accuracy

  # Train / test functions
  # ----------------------
  if params.last_layer_actication is None:
    seg_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    manifold_seg_loss = tf.keras.losses.KLDivergence(from_logits=True)
  else:
    seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    manifold_seg_loss = tf.keras.losses.KLDivergence()

  #@tf.function
  def train_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      if one_label_per_model:
        labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
        predictions = dnn_model(model_ftrs)
      else:
        labels = tf.reshape(labels_, (-1, sp[-2]))
        skip = params.min_seq_len
        predictions = dnn_model(model_ftrs)[:, skip:]
        labels = labels[:, skip + 1:]

      if params.train_loss == ['manifold_cros_entr']:
          labels = label_to_one_hot(labels=labels, params=params)
          manifold_seg_train_accuracy(labels, predictions)
          loss = manifold_seg_loss(labels, predictions)
      else:
        labels = label_to_one_hot(labels=labels, params=params)
        manifold_seg_train_accuracy(labels, predictions)
        loss = manifold_seg_loss(labels, predictions)
      loss += tf.reduce_sum(dnn_model.losses)

    train_logs['seg_loss'](loss)
    gradients = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))


    return loss

  def train_step_with_pred_vec(model_ftrs_, pred_vector_, labels, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      predictions = dnn_model(model_ftrs)
      if params.train_loss == ['manifold_cros_entr']:
          labels = label_to_one_hot(labels=labels, params=params)
          manifold_seg_train_accuracy(labels, predictions)
          loss = manifold_seg_loss(labels, predictions)
      else:
        seg_train_accuracy_one_hot = tf.keras.metrics.CategoricalAccuracy(name='seg_train_accuracy_one_hot')
        # seg_train_accuracy_one_hot(pred_vector_, predictions)
        manifold_seg_train_accuracy(labels, predictions)
        # if "MeshWalker" in config['dataset_path']:
        loss = manifold_seg_loss(pred_vector_, predictions)
        # loss = manifold_seg_loss(tf.nn.softmax(pred_vector_), predictions)
        # loss = manifold_seg_loss(pred_vector_, predictions)

      loss += tf.reduce_sum(dnn_model.losses)
    gradients = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))

    train_logs['seg_loss'](loss)

    return loss

  if params.train_loss == ['manifold_cros_entr']:
      test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
  else:
      test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  #@tf.function
  def test_step(model_ftrs_, labels_, one_label_per_model):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    if one_label_per_model:
      labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
      predictions = dnn_model(model_ftrs, training=False)
    else:
      labels = tf.reshape(labels_, (-1, sp[-2]))
      skip = params.min_seq_len
      if params.train_loss == ['manifold_cros_entr']:
          predictions = dnn_model(model_ftrs, training=False)[:, skip:]
      else:
          predictions = dnn_model(model_ftrs, training=False)[:, skip:]
      labels = labels[:, skip + 1:]

    if params.train_loss == ['manifold_cros_entr']:
          labels = label_to_one_hot(labels=labels, params=params)
    test_accuracy(labels, predictions)
    confusion = None
    return confusion
  # -------------------------------------
  # Loop over training EPOCHs
  # -------------------------
  one_label_per_model = params.one_label_per_model
  next_iter_to_log = 0
  e_time = 0
  accrcy_smoothed = tb_epoch = last_loss = None
  all_confusion = {}
  with tf.summary.create_file_writer(params.logdir).as_default():
    epoch = 0

    # start loop
    while optimizer.iterations.numpy() < params.iters_to_train + train_epoch_size * 2:
      epoch += 1
      if epoch % 10 == 0:
        print(params.logdir)
        print(config['description'])
      str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())

      # Save some logs & infos
      utils.save_model_if_needed(optimizer.iterations, dnn_model, params)

      if tb_epoch is not None:
        e_time = time.time() - tb_epoch
        tf.summary.scalar('time/one_epoch', e_time, step=optimizer.iterations)
        tf.summary.scalar('time/av_one_trn_itr', e_time / n_iters, step=optimizer.iterations)
        for name in time_msrs_names:
          if time_msrs[name]:  # if there is something to save
            tf.summary.scalar('time/' + name, time_msrs[name], step=optimizer.iterations)
            time_msrs[name] = 0
      tb_epoch = time.time()
      n_iters = 0
      tf.summary.scalar(name="train/learning_rate", data=optimizer._decayed_lr(tf.float32), step=optimizer.iterations)
      tf.summary.scalar(name="mem/free", data=utils.check_mem_and_exit_if_full(), step=optimizer.iterations)
      str_to_print += '; LR: ' + str(optimizer._decayed_lr(tf.float32).numpy())
      train_logs['seg_loss'].reset_states()
      tb = time.time()


      # training step
      for iter_db in range(train_epoch_size):
        for dataset_id in range(len(train_datasets)):

          if config['train_use_pred_vector']:
            name, model_ftrs, labels, pred_vector = train_ds_iters[dataset_id].next()
          else:
            name, model_ftrs, labels = train_ds_iters[dataset_id].next()

          dataset_type = utils.get_dataset_type_from_name(name)
          if params.learning_rate_dynamics != 'stable':
            utils.update_lerning_rate_in_optimizer(0, params.learning_rate_dynamics, optimizer, params)
          time_msrs['get_train_data'] += time.time() - tb
          n_iters += 1
          tb = time.time()


          if params.train_loss[dataset_id] == 'cros_entr':
            if config['train_use_pred_vector']:
              train_step_with_pred_vec(model_ftrs, pred_vector, labels, one_label_per_model=one_label_per_model)
              loss2show = 'seg_loss'
            else:
              train_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
              loss2show = 'seg_loss'
          elif params.train_loss[dataset_id] == 'manifold_cros_entr':
              train_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
              loss2show = 'seg_loss'
          else:
            raise Exception('Unsupported loss_type: ' + params.train_loss[dataset_id])
          time_msrs['train_step'] += time.time() - tb
          tb = time.time()
        if iter_db == train_epoch_size - 1:
          str_to_print += ', TrnLoss: ' + str(round(train_logs[loss2show].result().numpy(), 2))

      # Dump training info to tensorboard
      if optimizer.iterations >= next_iter_to_log:
        for k, v in train_logs.items():
          if v.count.numpy() > 0:
            tf.summary.scalar('train/' + k, v.result(), step=optimizer.iterations)
            v.reset_states()
        next_iter_to_log += params.log_freq
      # Run test on part of the test set
      if epoch % config['how_much_epoch_to_test_acc'] == 0 and test_dataset is not None:
        n_test_iters = 0
        tb = time.time()
        #for name, model_ftrs, labels in test_dataset:
        for i in range(n_tst_items):
          name, model_ftrs, labels = test_ds_iter.next()
          n_test_iters += model_ftrs.shape[0]
          if n_test_iters > params.n_models_per_test_epoch:
            break
          confusion = test_step(model_ftrs, labels, one_label_per_model=one_label_per_model)
          # Amir - added the case that confusion is none as a result of recon training
          if confusion is not None:
            dataset_type = utils.get_dataset_type_from_name(name)
            if dataset_type in all_confusion.keys():
              all_confusion[dataset_type] += confusion
            else:
              all_confusion[dataset_type] = confusion
        # Dump test info to tensorboard
        if accrcy_smoothed is None:
          accrcy_smoothed = test_accuracy.result()
        accrcy_smoothed = accrcy_smoothed * .9 + test_accuracy.result() * 0.1
        tf.summary.scalar('test/accuracy_' + dataset_type, test_accuracy.result(), step=optimizer.iterations)
        tf.summary.scalar('test/accuracy_smoothed', accrcy_smoothed, step=optimizer.iterations)
        str_to_print += ', test/accuracy_' + dataset_type + ': ' + str(round(test_accuracy.result().numpy(), 2))
        test_accuracy.reset_states()
        time_msrs['test'] += time.time() - tb

      str_to_print += ', time: ' + str(round(e_time, 1))
      print(str_to_print)


  return last_loss

def run_one_job(job, job_part, network_task):
  # Classifications
  job = job.lower()
  if job == 'modelnet40' or job == 'modelnet':
    params = params_setting.modelnet_params(network_task, config)

  if job == 'manifold40':
    params = params_setting.manifold_params(network_task, config)


  if job == 'shrec11':
    params = params_setting.shrec11_params(job_part, network_task, config)

  if job == 'cubes':
    params = params_setting.cubes_params(network_task, config)

  # Semantic Segmentations
  if job == 'human_seg':
    params = params_setting.human_seg_params(network_task, config)

  if job == 'coseg':
    params = params_setting.coseg_params(job_part, network_task, config)   #  job_part can be : 'aliens' or 'chairs' or 'vases'
  train_val(params)




if __name__ == '__main__':
  numpy.random.seed(0)
  utils.config_gpu()

  # get hyper params from yaml
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default = None, help='Path to the config file.')
  opts = parser.parse_args()
  config = utils.get_config(opts.config)
  job = config['job']
  job_part = config['job_part']
  # choose network task from: 'features_extraction', 'unsupervised_classification', 'semantic_segmentation', 'classification'. 'manifold_classification'
  network_task = config['network_task']
  run_one_job(job, job_part, network_task)