import os
from easydict import EasyDict
import numpy as np

from utils import utils
from data import dataset_prepare

if 0:
  MAX_AUGMENTATION = 90
  run_folder = 'runs_test'
elif 0:
  MAX_AUGMENTATION = 45
  run_folder = 'runs_aug_45'
else:
  MAX_AUGMENTATION = 360
  run_folder = 'runs_aug_360_must'

manifold_labels_type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}

model_net_modelnet40_labels = [None] * (max(manifold_labels_type_to_index_map.values()) + 1)
for key, value in manifold_labels_type_to_index_map.items():
  model_net_modelnet40_labels[value] = key
def use_pretrained_model(config, run_name):
  import json
  import tensorflow as tf
  with open(config['prev_model'] + '/params.txt') as fp:
    params = EasyDict(json.load(fp))
    params.net_start_from_prev_net = tf.train.latest_checkpoint(config['prev_model'])
    params.logdir = utils.get_run_folder(params.run_root_path + '/', '__' + run_name, params.cont_run_number)

    params.model_fn = params.logdir + '/learned_model.keras'
  return params


def set_up_default_params(network_task, run_name, cont_run_number=0, config = None):
  '''
  Define dafault parameters, commonly for many test case
  '''

  if config is not None:
    if config['use_prev_model'] is True:
      return use_pretrained_model(config, run_name)

  params = EasyDict()
  params.dataset = run_name
  params.cont_run_number = cont_run_number
  # params.run_root_path = os.path.expanduser('~') + '/mesh_walker/' + run_folder
  params.run_root_path = 'runs'+ '/mesh_walker/' + run_folder  #runs/runs_aug_360_must
  params.logdir = utils.get_run_folder(params.run_root_path + '/', '__' + run_name, params.cont_run_number)
  params.model_fn = params.logdir + '/learned_model.keras'

  # Optimizer params
  params.optimizer_type = 'cycle'  # sgd / adam / cycle
  params.learning_rate_dynamics = 'cycle'
  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 1e-4,
                                    'step_size': 10000})
  params.n_models_per_test_epoch = 300
  params.gradient_clip_th = 1

  # Dataset params
  params.classes_indices_to_use = None
  params.train_dataset_size_limit = np.inf
  params.test_dataset_size_limit = np.inf
  params.network_task = network_task
  params.normalize_model = True
  params.sub_mean_for_data_augmentation = True
  params.datasets2use = {}
  params.test_data_augmentation = {}
  params.train_data_augmentation = {}
  # params.aditional_network_params = []
  params.cut_walk_at_deadend = False

  params.network_tasks = [params.network_task]
  params.features_extraction = False
  if params.network_task == 'classification':
    params.n_walks_per_model = 1
    # Amir - changed to False to see what happens
    params.one_label_per_model = True
    params.train_loss = ['cros_entr']
    params.net = 'RnnWalkNet'
  elif params.network_task == 'manifold_classification':
    params.n_walks_per_model = 1
    params.one_label_per_model = True
    params.train_loss = ['manifold_cros_entr']
    params.net = 'Manifold_RnnWalkNet'
  elif params.network_task == 'semantic_segmentation':
    params.n_walks_per_model = 4
    params.one_label_per_model = False
    params.train_loss = ['cros_entr']
  elif params.network_task == 'unsupervised_classification':
    params.n_walks_per_model = 2
    params.one_label_per_model = True
    params.train_loss = ['triplet']
    params.net = 'Unsupervised_RnnWalkNet'
  elif params.network_task == 'features_extraction':
    params.n_walks_per_model = 2
    params.one_label_per_model = True
    params.train_loss = ['triplet']
  else:
    raise Exception('Unsupported params.network_task: ' + params.network_task)
  params.batch_size = int(config['max_batch_size'] / params.n_walks_per_model)



  # Other params
  params.log_freq = 1
  params.walk_alg = 'random_global_jumps'   # no_repeat / no_jumps / fast / fastest / only_jumps / local_jumps / no_local_jumps
  params.net_input = ['xyz'] # 'xyz', 'dxdydz', 'jump_indication'
  params.reverse_walk = False
  params.train_min_max_faces2use = [0, np.inf]
  params.test_min_max_faces2use = [0, np.inf]
  params.last_layer_actication = 'softmax'
  params.use_norm_layer = 'InstanceNorm' # BatchNorm / InstanceNorm / None
  params.layer_sizes = None

  params.initializers = 'orthogonal'
  params.adjust_vertical_model = False
  if config is not None and config['use_prev_model'] is True:
    import tensorflow as tf
    params.net_start_from_prev_net = tf.train.latest_checkpoint(config['prev_model'])
  else:
    params.net_start_from_prev_net = None


  params.net_gru_dropout = 0
  params.uniform_starting_point = False
  params.train_max_size_per_class = None    # None / 'uniform_as_max_class' / <a number>

  params.full_accuracy_test = None

  params.iters_to_train = 60e3



  return params

# Classifications
# ---------------
def modelnet_params(network_task, config=None):
  params = set_up_default_params(network_task, 'modelnet', 0, config)
  params.n_classes = 40

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 1e-5,
                                    'step_size': 10000})
  # p = 'modelnet40'
  params.train_min_max_faces2use = [0000, 4000]
  params.test_min_max_faces2use = [0000, 4000]
  ds_path = config['dataset_path']
  if len(ds_path) < 2:
    ds_path = '//datasets/datasets_processed/modelnet40_sortedlabel_f4000'
  params.datasets2use['train'] = [ds_path + '/train/train*.npz']
  params.datasets2use['test'] = [ds_path + '/test/test*.npz']

  params.seq_len = 2000
  params.min_seq_len = int(params.seq_len / 2)
  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.modelnet_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               'n_walks_per_model': 8,
                               }
  # params.full_accuracy_test = None
  params.n_walks_per_model = 1
  params.n_classes = 30
  params.seq_len = 200
  params.min_seq_len = int(params.seq_len / 2)
  params.last_layer_actication = 'softmax'
  # Parameters to recheck:
  params.iters_to_train = 1000e3
  params.net_input = ['xyz']
  params.walk_alg = 'no_jumps'   # no_jumps / global_jumps
  params.batch_size = 16
  if 0:
    params.iters_to_train = 2000e3
    params.net_input = ['dxdydz']

    params.last_layer_actication = None
    params.last_layer_activation = None
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6 / 2,
                                      'maximal_learning_rate': 0.0005 / 2,
                                      'step_size': 10000})
    params.net_start_from_prev_net = None
    params.batch_size = 16

  return params

def manifold_params(network_task, config=None):
  params = set_up_default_params(network_task, 'modelnet', 0, config)
  params.n_classes = 40
  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 1e-4,
                                    'step_size': 10000})
  params.train_min_max_faces2use = [0000, 500]
  params.test_min_max_faces2use = [0000, 500]
  params.batch_size = 128
  ds_path = config['dataset_path']
  if len(ds_path) < 2:
    ds_path = 'datasets_processed/walker_copycat_modelnet40'
  params.datasets2use['train'] = [ds_path + '/train/train*.npz']
  params.datasets2use['test'] = [ds_path + '/test/test*.npz']

  params.seq_len = 400
  params.min_seq_len = int(params.seq_len / 2)
  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': model_net_modelnet40_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               }
  # params.full_accuracy_test = None

  # Parameters to recheck:
  params.iters_to_train = 1000e3
  params.net_input = ['xyz']
  params.walk_alg = 'no_jumps'   # no_jumps / global_jumps

  if 0:
    params.iters_to_train = 2000e3
    params.net_input = ['dxdydz']

    params.last_layer_actication = None
    params.last_layer_activation = None
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6 / 2,
                                      'maximal_learning_rate': 0.0005 / 2,
                                      'step_size': 10000})
    params.net_start_from_prev_net = None
    params.batch_size = 16

  return params
def mesh_net_params(network_task, config=None):
  params = set_up_default_params(network_task, 'mesh_net', 0, config)
  params.n_classes = 40

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 0.0005,
                                    'step_size': 10000})

  p = 'mesh_net'
  params.train_min_max_faces2use = [0000, 4000]
  params.test_min_max_faces2use = [0000, 4000]
  # params.train_data_augmentation = {'vertex_gaussian_noisy': (0, 0.01)}

  ds_path = config['dataset_path']
  if len(ds_path) < 2:
    ds_path = 'datasets_processed/mesh_net_modelnet40'
  params.datasets2use['train'] = [ds_path + '/*train*.npz']
  params.datasets2use['test'] = [ds_path + '/*test*.npz']

  params.seq_len = 400
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               }


  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['xyz']
  params.walk_alg = 'random_global_jumps'   # no_jumps / global_jumps

  if 1:
    params.iters_to_train = 2000e3
    params.net_input = ['xyz']

    params.last_layer_actication = None
    params.last_layer_activation = None
    params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6 / 2,
                                      'maximal_learning_rate': 0.0005 / 2,
                                      'step_size': 10000})

    params.net_start_from_prev_net = None
    params.batch_size = 16

  return params

def cubes_params(network_task, config = None):
  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100
  params = set_up_default_params(network_task, 'cubes', 0, config)
  params.n_classes = 22
  params.seq_len = 100
  params.min_seq_len = int(params.seq_len / 2)

  p = 'cubes'
  params.datasets2use['train'] = [os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + p + '/*train*.npz']
  params.datasets2use['test'] = [os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + p + '/*test*.npz']

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.cubes_labels,
                               }

  params.iters_to_train = 460e3

  return params

def shrec11_params(split_part, network_task, config = None):
  # split_part is one of the following:
  # 10-10_A / 10-10_B / 10-10_C
  # 16-04_a / 16-04_b / 16-04_C

  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100

  params = set_up_default_params(network_task, 'shrec11_' + split_part, 0, config)
  params.n_walks_per_model = 1
  params.n_classes = 30
  params.seq_len = 200
  params.min_seq_len = int(params.seq_len / 2)
  params.last_layer_actication = 'softmax'
  # params.optimizer_type = 'adam'
  # params.learning_rate = [1e-6]
  params.net = config['network_arch']

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 5e-6,
                                    'maximal_learning_rate': 5e-5,
                                    'step_size': 10000})
  params.batch_size = 240
  ds_path = config['dataset_path']
  if len(ds_path) < 2:
    ds_path = config['dataset_path']
  params.datasets2use['train'] = [ds_path + '/train/*.npz']
  params.datasets2use['test'] = [ds_path + '/test/*.npz']

  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.shrec11_labels}
  params.net_using_from_prev_net = None
  params.iters_to_train = 10e5

  if 'last_layer_activation' in config:
    params.last_layer_activation = config['last_layer_activation']
  else:
    params.last_layer_activation = 'softmax'

  return params


def few_class_params(network_task, config=None):
  # split_part is one of the following:
  # 10-10_A / 10-10_B / 10-10_C
  # 16-04_a / 16-04_b / 16-04_C

  # |V| = 250 , |F| = 500 => seq_len = |V| / 2.5 = 100

  params = set_up_default_params(network_task, 'few_calss_', 0, config)
  params.n_walks_per_model = 1
  params.n_classes = 30
  params.seq_len = 2000
  params.min_seq_len = int(params.seq_len / 2)
  params.last_layer_actication = 'softmax'
  # params.optimizer_type = 'adam'
  # params.learning_rate = [1e-6]
  params.net = config['network_arch']

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 5e-6,
                                    'maximal_learning_rate': 5e-5,
                                    'step_size': 10000})
  params.batch_size = 1
  ds_path = config['dataset_path']
  if len(ds_path) < 2:
    ds_path = config['dataset_path']
  params.datasets2use['train'] = [ds_path + '/train/*.npz']
  params.datasets2use['test'] = [ds_path + '/test/*.npz']

  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'labels': dataset_prepare.shrec11_labels}
  params.net_using_from_prev_net = None
  params.iters_to_train = 10e5

  if 'last_layer_activation' in config:
    params.last_layer_activation = config['last_layer_activation']
  else:
    params.last_layer_activation = 'softmax'

  return params


# Semantic Segmentation
# ---------------------
def human_seg_params(network_task, config = None):
  # |V| = 750 , |F| = 1500 => seq_len = |V| / 2.5 = 300
  params = set_up_default_params(network_task, 'human_seg', 0, config)
  params.n_classes = 9
  params.seq_len = 300

  if 1: # MeshCNN data
    sub_dir = 'human_seg_from_meshcnn'
  if 0: # Simplification to 1.5k facesCUBES
    sub_dir = 'sig17_seg_benchmark-1.5k'
  if 0: # Simplification to 4k faces 4000 / 2 / 2.5 = 800
    sub_dir = 'sig17_seg_benchmark-4k'
    params.seq_len = 1200
  if 0: # Simplification to 6k faces 6000 / 2 / 2.5 = 1200
    sub_dir = 'sig17_seg_benchmark-6k'
    params.seq_len = 2000
  if 0: # Simplification to 8k faces
    sub_dir = 'sig17_seg_benchmark-8k'
    params.seq_len = 1600
    params.batch_size = int(16 / params.n_walks_per_model)
  if 0:
    params.n_target_vrt_to_norm_walk = 3000
    sub_dir = 'sig17_seg_benchmark-no_simplification'
    params.seq_len = 2000
  p = os.path.expanduser('~') + '/mesh_walker/datasets_processed/' + sub_dir + '/'
  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  params.min_seq_len = int(params.seq_len / 2)
  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'n_iters': 32}


  # Parameters to recheck:
  params.iters_to_train = 100e3


  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 2e-5,
                                    'step_size': 10000})

  return params


def coseg_params(type, network_task, config = None): # aliens / chairs / vases
  # |V| = 750 , |F| = 1500 => seq_len = |V| / 2.5 = 300
  sub_folder = 'coseg_' + type
  p = os.path.expanduser('~') + '/mesh_walker/datasets_processed/coseg_from_meshcnn/' + sub_folder + '/'
  params = set_up_default_params(network_task, 'coseg_' + type, 0, config)
  params.n_classes = 10
  params.seq_len = 300
  params.min_seq_len = int(params.seq_len / 2)

  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  params.iters_to_train = 200e3
  params.train_data_augmentation = {'rotation': MAX_AUGMENTATION}

  params.full_accuracy_test = {'dataset_folder': params.datasets2use['test'][0],
                               'n_iters': 32}


  return params


