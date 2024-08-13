import argparse
from utils import utils
import numpy as np
import attack_single_mesh
from surrogate_train import params_setting

#All data sets paths
meshCnn_and_Pd_meshNet_shrec_path ='datasets_processed/meshCNN_and_PD_meshNet_source_data/'
meshCnn_shrec_vertices_and_faces = 'datasets_processed/meshCNN_faces_vertices_labels/'
pd_MeshNet_shrec_vertices_and_faces = 'datasets_processed/Pd_meshNet_faces_vertices_labels/'
meshWalker_model_net_path = 'datasets_processed/walker_copycat_modelnet40/'
mesh_net_path = 'datasets_processed/mesh_net_modelnet40/'
meshWalker_cubes_path = '/home/kang/SSD/datasets/random-walks/datasets_processed/cubes/'
mesh_net_labels = ['night_stand', 'range_hood', 'plant', 'chair', 'tent',
    'curtain', 'piano', 'dresser', 'desk', 'bed',
    'sink',  'laptop', 'flower_pot', 'car', 'stool',
    'vase', 'monitor', 'airplane', 'stairs', 'glass_box',
    'bottle', 'guitar', 'cone',  'toilet', 'bathtub',
    'wardrobe', 'radio',  'person', 'xbox', 'bowl',
    'cup', 'door',  'tv_stand',  'mantel', 'sofa',
    'keyboard', 'bookshelf',  'bench', 'table', 'lamp']
walker_shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]
# walker_shrec11_labels.sort()
walker_model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]

meshCNN_and_Pd_meshNet_shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]
meshCNN_and_Pd_meshNet_shrec11_labels.sort()
shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]

def get_dataset_path(config = None):
    if config is None:
        exit("Your configuration file is None... Exiting")
    if config['arch'] == 'WALKER' and config['dataset'] == 'MODELNET40':
      # config['trained_model'] = 'trained_models/walker_modelnet_imitating_network'
      return meshWalker_model_net_path
    elif config['arch'] == 'MESHNET' and config['dataset'] == 'MODELNET40':
      # config['trained_model'] = 'trained_models/mesh_net_imitating_network'
      return mesh_net_path
    if config['arch'] == 'WALKER' and config['dataset'] == 'SHREC11':
      #config['trained_model'] = 'trained_models/walker_shrec11_imitating_network'
      return config['dataset_to_be_attack']
    elif config['arch'] == 'MESHCNN' and config['dataset'] == 'SHREC11':
      # config['trained_model'] = 'trained_models/meshCNN_imitating_network'
      return meshCnn_shrec_vertices_and_faces
    elif config['arch'] == 'PDMESHNET' and config['dataset'] == 'SHREC11':
      # config['trained_model'] = 'trained_models/pd_meshnet_imitating_network'
      return pd_MeshNet_shrec_vertices_and_faces
    elif config['arch'] == 'WALKER' and config['dataset'] == 'CUBES':
      # config['trained_model'] = 'trained_models/walker_cubes_imitating_network'
      return meshWalker_cubes_path
    else:
      exit("Please provide a valid dataset name in recon file.")


def attack_mesh_net_models(config=None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 40):
        config['source_label'] = i
        name_of_class = mesh_net_labels[config['source_label']]
        model_net_files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in model_net_files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in model_net_files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            name_parts = [name for name in name_parts if name.isnumeric()]
            id = name_parts[0] #name_parts[1] + '_' + name_parts[2] + '_' + name_parts[-1][:-4]
            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id, labels=mesh_net_labels)

    return


def attack_walker_model_net_models(config=None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 40):
        config['source_label'] = i
        name_of_class = walker_model_net_labels[config['source_label']]
        model_net_files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in model_net_files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            num_of_models = [name for name in model_net_files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            name_parts = re.split(pattern='_', string=model_name)
            name_parts[-1] = name_parts[-1][:-4]
            id = ''
            for i in range(len(name_parts)):
                if name_parts[i].isdigit():
                    id = id +'_' + name_parts[i]

            _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path+model_name, id=id, labels=walker_model_net_labels)

    return meshCNN_and_Pd_meshNet_shrec11_labels


def attack_meshCNN_shrec11_models(config = None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)
    # for name_of_class in meshCNN_and_Pd_meshNet_shrec11_labels:
    for i in range(0, 30):
        name_of_class = meshCNN_and_Pd_meshNet_shrec11_labels[i]
        config['source_label'] = meshCNN_and_Pd_meshNet_shrec11_labels.index(name_of_class)
        if name_of_class == 'man':
            files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test')
                               and file.__contains__(name_of_class) and not file.__contains__('woman')]
        else:
            files_to_attack = [file for file in os.listdir(path=dataset_path) if
                               file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue

            num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue

            name_parts = re.split(pattern='_', string=model_name)
            # 'two_balls' and 'dino_ske' contain '_'
            if name_of_class == 'two_balls' or name_of_class == 'dino_ske':
                id = name_parts[3]
                _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
                                                               datasets_labels=walker_shrec11_labels)
            elif name_of_class != 'two_balls' and name_of_class != 'dino_ske':
                id = name_parts[2]
                _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
                                                               datasets_labels=walker_shrec11_labels)
    return


import re

import os


def attack_shrec11_f4000_specify_id(config=None):
    id = config['id']
    class_name = config['class_name']
    label_to_use = walker_shrec11_labels
    if config is None or id is None or class_name is None:
        return
    dataset_path = get_dataset_path(config=config)
    config['source_label'] = label_to_use.index(class_name)

    # 根据给定的类名和ID构造文件名
    expected_filename = f"{id}_simplified_to_4000.npz"
    expected_filepath = os.path.join(dataset_path, expected_filename)

    # 检查文件是否存在
    if os.path.isfile(expected_filepath):
        _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=expected_filepath, id=id,
                                                       datasets_labels=label_to_use)
    else:
        print(f"File {expected_filename} does not exist in the dataset path.")

    return
def attack_manifold_models(config = None):
    if config is None:
        return
    dataset_path = config['dataset_to_be_attack']
    manifold40_labels = params_setting.model_net_modelnet40_labels
    # for name_of_class in meshCNN_and_Pd_meshNet_shrec11_labels:
    for i in range(14, 20):
        name_of_class = manifold40_labels[i]
        config['source_label'] = manifold40_labels.index(name_of_class)

        files_to_attack = [file for file in os.listdir(path=dataset_path) if
                           file.__contains__('test') and file.__contains__(name_of_class)]

        for model_name in files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue

            num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue

            name_parts = re.split(pattern='_', string=model_name)
            # 'two_balls' and 'dino_ske' contain '_'
            if name_of_class.__contains__('_'):
                id = name_parts[5]
                _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
                                                               datasets_labels=manifold40_labels)
            else:
                id = name_parts[3]
                _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
                                                               datasets_labels=manifold40_labels)
    return




def attack_walker_shrec11_models(config = None):
    if config is None:
        return
    dataset_path = get_dataset_path(config=config)

    for i in range(0, 30):
        config['source_label'] = i
        name_of_class = walker_shrec11_labels[config['source_label']]

        #取对应标签的类名 name_of_class
        #代码列出了目标文件夹下的所有文件，并筛选出符合特定条件的文件，存储在 files_to_attack 列表中。条件包括文件名包含 "test" 和包含 name_of_class
        #这里之前作者的代码有小bug，因为shrec11里即有woman种类也有man种类，'woman'包含'man'
        #所以遍历到name_of_class = man时，会把所有woman也加载进去
        if name_of_class == 'man':
            files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test')
                               and file.__contains__(name_of_class) and not file.__contains__('woman')]
        else:
            files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test') and file.__contains__(name_of_class)]


        for model_name in files_to_attack:
            if str(model_name).__contains__('attacked'):
                continue
            #如果存在多个文件名以相同的模型名称开头（model_name[0:-4]），则跳过，可能是多个版本的相同模型
            num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
            if len(num_of_models) > 1:
                continue
            
            name_parts = re.split(pattern='_', string=model_name)
            # 这里加上判断语句，because 'two_balls' and 'dino_ske' contain '_'
            if name_of_class == 'two_balls' or name_of_class == 'dino_ske':
                id =name_parts[3]
                _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
                                                          labels=walker_shrec11_labels)
            elif name_of_class != 'two_balls' and name_of_class != 'dino_ske':
                id = name_parts[2]
                _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
                                                      labels=walker_shrec11_labels)
    return


# def attack_single_walker_shrec11_models(config=None):
#     if config is None:
#         return
#     dataset_path = "/home/kang/SSD/Projects/Random-Walks-for-Adversarial-Meshes/datasets/datasets_raw/test_scale_processed/test/test_centaur_scaled_centaur_565_attacked_not_changed_3999.npz"
#
#     for i in range(8, 30):
#         config['source_label'] = i
#         name_of_class = walker_shrec11_labels[config['source_label']]
#
#         # 取对应标签的类名 name_of_class
#         # 代码列出了目标文件夹下的所有文件，并筛选出符合特定条件的文件，存储在 files_to_attack 列表中。条件包括文件名包含 "test" 和包含 name_of_class
#         # 这里之前作者的代码有小bug，因为shrec11里即有woman种类也有man种类，'woman'包含'man'
#         # 所以遍历到name_of_class = man时，会把所有woman也加载进去
#         if name_of_class == 'man':
#             files_to_attack = [file for file in os.listdir(path=dataset_path) if file.__contains__('test')
#                                and file.__contains__(name_of_class) and not file.__contains__('woman')]
#         else:
#             files_to_attack = [file for file in os.listdir(path=dataset_path) if
#                                file.__contains__('test') and file.__contains__(name_of_class)]
#
#         for model_name in files_to_attack:
#             if str(model_name).__contains__('attacked'):
#                 continue
#             # 如果存在多个文件名以相同的模型名称开头（model_name[0:-4]），则跳过，可能是多个版本的相同模型
#             num_of_models = [name for name in files_to_attack if name.__contains__(model_name[0:-4])]
#             if len(num_of_models) > 1:
#                 continue
#
#             name_parts = re.split(pattern='_', string=model_name)
#             # 这里加上判断语句，because 'two_balls' and 'dino_ske' contain '_'
#             if name_of_class == 'two_balls' or name_of_class == 'dino_ske':
#                 id = name_parts[3]
#                 _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
#                                                           labels=shrec11_labels)
#             elif name_of_class != 'two_balls' and name_of_class != 'dino_ske':
#                 id = name_parts[2]
#                 _ = attack_single_mesh.attack_single_mesh(config=config, source_mesh=dataset_path + model_name, id=id,
#                                                           labels=shrec11_labels)
#     return



def main():
    # get hyper params from yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to the config file.')
    opts = parser.parse_args()
    config = utils.get_config(opts.config)
    np.random.seed(0)
    utils.config_gpu(-1)
    if config['gpu_to_use'] >= 0:
        utils.set_single_gpu(config['gpu_to_use'])
    # attack_manifold_models(config=config)
    attack_meshCNN_shrec11_models(config=config)
    # attack_shrec11_f4000_specify_id(config=config)
    # attack_Pd_meshNet_shrec11_models(config=config)
    # attack_ExMeshCNN_shrec11_models(config=config)
    # attack_walker_shrec11_models(config=config)
    # attack_single_walker_shrec11_models(config= config)
    # attack_walker_model_net_models(config=config)
    # attack_mesh_net_models(config=config)
    return 0  


if __name__ == '__main__':
    main()
