import argparse
from vertices_clustering import vertices_clustering_via_GMM,vertices_clustering_via_KNN
from data import dataset_prepare_utils
from utils import utils
import copy
import open3d as o3d
from models.perceptual_loss import get_perceptual_loss, threshold_loss
from models import beyesican_rnn_model, rnn_model, perceptual_loss
# import igl
from attack import bound


def compute_area_weighted_vertex_normals(vertices, faces, face_normals):

    # num_vertices = len(vertices)
    num_vertices = tf.shape(vertices)[0]
    vertex_normals = tf.zeros_like(vertices, dtype=tf.float32)
    # vertex_normals = np.zeros_like(vertices, dtype=float)
    for face in tf.unstack(faces):

        A, B, C = tf.gather(vertices, face)
        # 计算两个边的向量
        edge1 = B - A
        edge2 = C - A
        # 计算三角形法线
        normal = tf.linalg.cross(edge1, edge2)

        # 计算三角形面积（使用法线长度的一半）
        area = 0.5 * tf.norm(normal)
        # 面积加权，累加到每个顶点
        vertex_normals_update = tf.tensor_scatter_nd_add(vertex_normals, tf.expand_dims(face, axis=1), area * normal)
        # 更新顶点法线tensor
        vertex_normals = vertex_normals_update

        # 归一化顶点法线（注意：需要避免除以0，可以添加一个小的epsilon值）
    norm = tf.norm(vertex_normals, axis=1, keepdims=True)
    epsilon = tf.keras.backend.epsilon()  # 使用Keras的epsilon值避免除以0
    norm = tf.maximum(norm, epsilon)  # 保证norm不会为0
    vertex_normals = vertex_normals / norm
    # mesh['vertex_normal'] = vertex_normals
    return vertex_normals

def covariances_normalize(covariances):
    sum = tf.reduce_sum(covariances)
    covariances_normalized = covariances / sum
    covariances_normalized = tf.cast(covariances_normalized, tf.float32)
    return covariances_normalized
def prepare_normal_per_face(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  face_vertices = vertices[faces] # 每个面的顶点坐标
  # 计算每个面的两个边向量
  edge1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
  edge2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
  # 计算法向量
  cross_product = np.cross(edge1, edge2)
  # 归一化法向量
  normal_vectors = cross_product / np.linalg.norm(cross_product, axis=1)[:, np.newaxis]
  # mesh['face_normal'] = normal_vectors
  return normal_vectors
def determine_class_attack_weights(covariances):
    '''
        通过方差来制定攻击权重
    '''
    covariances = tf.convert_to_tensor(covariances,dtype=tf.float32)
    # covariances = covariances_normalize(covariances)

    # 找到最大元素的位置
    # max_index = np.argmax(covariances)
    # 创建一个形状为 (K, 1) 的数组，初始值为 0
    # weights = np.zeros_like(covariances)
    # weights = np.full_like(covariances,1)
    # weights = tf.keras.activations.hard_sigmoid(covariances)
    # weights = tf.keras.activations.sigmoid(covariances)
    weights = tf.ones_like(covariances,dtype=tf.float32)
    # weights = covariances * 20
    # weights = covariances
    return weights

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_labels.sort()
# get hyper params from yaml
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='recon_config.yaml', help='Path to the config file.')
opts = parser.parse_args()
config = utils.get_config(opts.config)

if config['gpu_to_use'] >= 0:
    utils.set_single_gpu(config['gpu_to_use'])

import os, shutil, time
from easydict import EasyDict
import json
import cv2
import numpy as np
import tensorflow as tf
import pyvista as pv
import pylab as plt
from src.surrogate_train import dataset
import src.data_process.dataset_prepare


def dump_mesh(mesh_data, path, cpos, iter, x_server_exists):
    """
    Saves a picture of the mesh
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    if x_server_exists:
        window_size = [512, 512]
        p = pv.Plotter(off_screen=1, window_size=(int(window_size[0]), int(window_size[1])))
        faces = np.hstack([[3] + f.tolist() for f in mesh_data['faces']])
        surf = pv.PolyData(mesh_data['vertices'], faces)
        p.add_mesh(surf, show_edges=False, color=None)
        p.camera_position = cpos
        # p.set_background("#AAAAAA", top="White")
        p.set_background("#AAAAAA")
        rendered = p.screenshot()
        p.close()
        img = rendered.copy()
        my_text = str(iter)
        cv2.putText(img, my_text, (img.shape[1] - 100, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 255), thickness=2)
        cv2.imwrite(path + '/img_' + str(dump_mesh.i).zfill(5) + '.jpg', img)
    dump_mesh.i += 1

dump_mesh.i = 0

def save_last_class_weight(class_weight, save_path):
    np.save(save_path, class_weight)
    return

def deform_add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dump_model=True):
    """
    Saves the new attacked mesh model
    """
    m = {}
    for k, v in mesh_data.items():
        if k in fileds_needed:
            m[k] = v
    for field in fileds_needed:
        if field not in m.keys():
            if field == 'labels_fuzzy':
                m[field] = np.zeros((0,))
            if field == 'walk_cache':
                m[field] = np.zeros((0,))
            if field == 'kdtree_query':
                dataset_preprocess.prepare_edges_and_kdtree(m)

    if dump_model:
        np.savez(out_fn, **m)

# def get_res_path(config, id=-1, labels=None):
#     """
#     Sets the result path in which the mesh and its pictures are saved
#     """
#     if labels is None:
#         exit("Error, no shrec labels")
#
#     net_name = config['trained_model'].split('/')[-1]
#     if len(net_name) > 0:
#         res_path = 'attacks/' + net_name + '/' + labels[config['source_label']]
#     else:
#         res_path = 'attacks/' + labels[config['source_label']]
#     if id != -1:
#         res_path += '_' + str(id)
#     return res_path, net_name
def get_res_path(config, id = -1, labels = None):
  """
  Sets the result path in which the mesh and its pictures are saved
  """
  if labels is None:
    exit("Error, no shrec labels")

  net_name = config['trained_model'].split('/')[-1]
  if len(net_name) > 0:
    # res_path = 'attacks/' + net_name +'/' + labels[config['source_label']]
    res_path = config['path_to_save_attack_results'] +'/' + labels[config['source_label']]
  else:
    res_path =  config['path_to_save_attack_results'] +'/' + labels[config['source_label']]
  if id != -1:
    res_path+= '_'+str(id)
  return res_path, net_name

def plot_preditions(params, dnn_model, config, mesh_data, result_path, num_iter, x_axis, source_pred_list):
    """
    Saves as image a graph of the prediction of the network on the changed mesh
    """
    params.n_walks_per_model = 16
    features, labels = dataset.mesh_data_to_walk_features_modified(mesh_data = mesh_data, dataset_params=params)
    ftrs = tf.cast(features[:, :, :3], tf.float32)
    eight_pred = dnn_model(ftrs, classify=True, training=False)
    sum_pred = tf.reduce_sum(eight_pred, 0)
    print("source_label number ", config['source_label'], " over " + str(params.n_walks_per_model) + " runs is: ",
          (sum_pred.numpy())[config['source_label']] / params.n_walks_per_model)
    source_pred_list.append((sum_pred.numpy())[config['source_label']] / params.n_walks_per_model)
    params.n_walks_per_model = 8

    if not os.path.isdir(result_path + '/plots/'):
        os.makedirs(result_path + '/plots/')
    # plot the predictions
    x_axis.append(num_iter)

    plt.plot(x_axis, source_pred_list)
    plt.title(str(config['source_label']) + ": source pred")
    plt.savefig(result_path + '/plots/' + 'source_pred.png')
    plt.close()
    return

def define_network_and_its_params(config=None):
    """
    Defining the parameters of the network, called params, and loads the trained model, called dnn_model
    """
    # 加载params.txt中的参数
    with open(config['trained_model'] + '/params.txt') as fp:
        params = EasyDict(json.load(fp))
    logdir = config['trained_model']
    import glob
    print(logdir)
    # print(glob.glob(logdir + "/*.keras"))
    model_fn = glob.glob(logdir + "/" + "learned_model2keep*.keras")[-1]
    # Define network parameters
    params.batch_size = 1
    params.seq_len = config['walk_len']
    params.n_walks_per_model = config['num_walks_per_iter']
    params.set_seq_len_by_n_faces = False
    params.data_augmentaion_vertices_functions = []
    params.label_per_step = False
    params.n_target_vrt_to_norm_walk = 0
    params.net_input += ['vertex_indices']
    dataset.setup_features_params(params, params)
    dataset.mesh_data_to_walk_features.SET_SEED_WALK = False
    try:
        if (config["network_arch"] == 'RnnWalkNet'):
            dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, 3, model_fn,
                                             model_must_be_load=True, dump_model_visualization=False)
        elif (config["network_arch"] == "RnnMixtureNet"):
            dnn_model = beyesican_rnn_model.RnnMixtureNet(params, params.n_classes, 3, model_fn,
                                                          model_must_be_load=True, dump_model_visualization=False)
    except NameError:
        print("network arch error")

    return params, dnn_model

def create_triangle_mesh(vertices, neighbor_indices):
    flat_indices = vertices.flatten()
    valid_indices = flat_indices[flat_indices != -1]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_indices).reshape(-1, 3))
    o3d.io.write_triangle_mesh("output_mesh.obj", mesh)
    return

def prepare_normal_per_face(mesh):
    vertices = mesh['vertices']
    faces = mesh['faces']
    face_vertices = vertices[faces] # 每个面的顶点坐标
    # 计算每个面的两个边向量
    edge1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    edge2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
    # 计算法向量
    cross_product = np.cross(edge1, edge2)
    # 归一化法向量
    normal_vectors = cross_product / np.linalg.norm(cross_product, axis=1)[:, np.newaxis]
    mesh['face_normal'] = normal_vectors
    return normal_vectors
def extract_and_save_geometric_infos_from_mesh_data(mesh_data):
    edge_pairs = dataset_prepare_utils.compute_edge_pairs(mesh_data)
    mesh_data["edge_pairs"] = edge_pairs
    vertex_normals = dataset_prepare_utils.compute_area_weighted_vertex_normals(mesh_data)
    mesh_data['vertex_normal'] = vertex_normals
    face_normals = dataset_prepare_utils.prepare_normal_per_face(mesh_data)
    mesh_data['face_normal'] = face_normals
    return edge_pairs, vertex_normals, face_normals

def extract_spatial_infos_from_mesh_data(mesh_data):
    # mesh_vertices_np = mesh_data['vertices']
    # mesh_faces_np = mesh_data['faces']
    # mesh_edges_np = mesh_data['edges']
    mesh_vertices_tensor = tf.convert_to_tensor(mesh_data['vertices'],dtype=tf.float32)
    mesh_faces_tensor = tf.convert_to_tensor(mesh_data['faces'], dtype=tf.int32)
    mesh_edges_tensor = tf.convert_to_tensor(mesh_data['edges'])
    return  mesh_vertices_tensor, mesh_faces_tensor, mesh_edges_tensor

def attack_single_mesh(config=None, source_mesh=None, id=-1, datasets_labels=None):

    if datasets_labels is None or config is None:
        exit(-1)

    # Defining network's parameters and model
    network_params, network_dnn_model = define_network_and_its_params(config = config)

    # Defining output path
    result_path, net_name = get_res_path(config = config, id = id, labels = datasets_labels)

    # print some infos, source label and output dir of current mesh
    print("source label: ", config['source_label'], " output dir: ", result_path)

    # if use_last == False, we erase previous attack results
    if os.path.isdir(result_path) and config['use_last'] is False:
        shutil.rmtree(result_path)

    # Defining original mesh data - Either use the last saved in the folder or the original one
    orig_mesh_data_path = source_mesh
    # we need to load the clean mesh (do not disturbed yet) to compute perceptual loss to constrain the distortion
    clean_mesh_data =  np.load(source_mesh, encoding='latin1', allow_pickle=True)
    if config['use_last'] is True:
        if os.path.exists(result_path + '/' + 'last_model.npz'):  # A previous model exists
            orig_mesh_data_path = result_path + '/last_model.npz'
        elif os.path.exists(source_mesh[0:-4] + '_' + str(id) + '_attacked.npz'):  # A previous attacked model exists
            orig_mesh_data_path = source_mesh[0:-4] + '_' + str(id) + '_attacked.npz'

    # orig_mesh may from last attack or may from a clean one
    orig_mesh_data = np.load(orig_mesh_data_path, encoding='latin1', allow_pickle=True)
    # load mesh data
    mesh_clean = {k: v for k, v in clean_mesh_data.items()}
    mesh_clean_deep_copy = copy.deepcopy(mesh_clean)
    mesh_data = {k: v for k, v in orig_mesh_data.items()}
    mesh_to_save = {k: v for k, v in clean_mesh_data.items()}
    edge_pairs, vertex_normals, face_normals = extract_and_save_geometric_infos_from_mesh_data(mesh_data)
    edge_pairs_clean, vertex_normals_clean, face_normals_clean = extract_and_save_geometric_infos_from_mesh_data(mesh_clean)
    edge_pairs_clean, vertex_normals_clean, face_normals_clean = extract_and_save_geometric_infos_from_mesh_data(mesh_clean_deep_copy)
    mesh_vertices, mesh_faces, mesh_edges = extract_spatial_infos_from_mesh_data(mesh_data)
    mesh_vertices_clean, mesh_faces_clean, mesh_edges_clean = extract_spatial_infos_from_mesh_data(mesh_clean)

    if config['cluster_type'] == 'GMM':
        means, covariances, cluster_labels = vertices_clustering_via_GMM(mesh_clean_deep_copy, config['K'], result_path)
    elif config['cluster_type'] == 'KNN':
        cluster_labels = vertices_clustering_via_KNN(mesh_clean_deep_copy, config['K'], result_path)
    # mesh_orig_vertices = tf.convert_to_tensor(mesh_orig['vertices'],dtype=tf.float32)
    # 将输入的数据都转成numpy数组
    mesh_vertices_normalized = dataset.norm_model(mesh_vertices)
    mesh_face_normals = tf.convert_to_tensor(mesh_data['face_normal'], dtype=tf.float32)
    mesh_vertices_normalized = tf.convert_to_tensor(mesh_vertices_normalized, dtype=tf.float32)
    mesh_edge_pairs = tf.convert_to_tensor(mesh_data['edge_pairs'],dtype=tf.int64)
    mesh_edges = tf.convert_to_tensor(mesh_data['edges'], dtype=tf.int32)
    mesh_vertices = tf.Variable(mesh_vertices, trainable=False, dtype=tf.float32)
    loss_weights = config['regularization_loss_weights']
    # 初始化一个二维列表，用于存储每个聚类类别的数据

    # 将数据竖直得组合成tensor，需要时打开
    # tensor_mesh_vertices = [np.vstack(cluster) for cluster in clustered_mesh_vertices_arrays]
    # 对聚类的方差进行排序，决定攻击时不同类别扰动的权重
    # 根据聚类的方差大小来决定每个类别的攻击权重
    fields_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'labels_fuzzy']
    if config['cluster_type'] == 'GMM':
        class_weights_0 = determine_class_attack_weights(covariances)
    elif config['cluster_type'] == 'KNN':
        class_weights_0 =  np.ones(config['K'])

    misclassified_file_path = os.path.join(result_path, "misclassified.txt")
    last_class_weights_file_path = os.path.join(result_path, "last_class_weights.npy")

    if config['use_last'] is True and os.path.isfile(last_class_weights_file_path):
        class_weights = np.load(last_class_weights_file_path)
        class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    elif config['cluster_type'] == 'GMM':
        class_weights = determine_class_attack_weights(covariances)
    elif config['cluster_type'] == 'KNN':
        class_weights = np.ones(config['K'])

    if config['use_last'] is True and os.path.isfile(misclassified_file_path):
        print(f"{result_path.split('/')[-1]} already successfully attacked.")
        for curr_perceptual_bound in config['perceptual_bound']:
            curr_need = os.path.join(result_path, "last_model_perceptual_bound_" + str(curr_perceptual_bound) + '.npz')
            if os.path.exists(curr_need) and os.path.isfile(curr_need):
                continue
            deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                             out_fn=curr_need)
        # for curr_adaptive_perceptual_bound in config['adaptive_perceptual_bound']:
        #     curr_need = os.path.join(result_path, "last_model_adaptive_perceptual_bound_" + str(
        #         curr_adaptive_perceptual_bound) + '.npz')
        #     if os.path.exists(curr_need) and os.path.isfile(curr_need):
        #         continue
        #     deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
        #                                      out_fn=curr_need)
        for curr_l2_bound in config['L2_bound']:
            curr_need =  os.path.join(result_path, "L2_bound_" + str(curr_l2_bound) + '.npz')
            if os.path.exists(curr_need) and os.path.isfile(curr_need):
                continue
            deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                                 out_fn=curr_need)
        for curr_norm_bound in config['norm_bound']:
            curr_need =  os.path.join(result_path, "norm_bound_" + str(curr_norm_bound) + '.npz')
            if os.path.exists(curr_need) and os.path.isfile(curr_need):
                continue
            deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                                 out_fn=curr_need)
        print("do not perturb more")
        return
    # 根据每个顶点的类别  以及每个类别的攻击权重，来确定顶点的攻击权重
    # # vertex_attack_weights = class_weights[cluster_labels]
    # vertex_attack_weights = tf.gather(class_weights,cluster_labels)

    # vertex_attack_weights = tf.Variable(vertex_attack_weights, trainable=True)
    # Defining parameters that keep track of the changes
    loss = []
    cpos = None
    last_dev_res = 0
    last_plt_res = 0

    source_pred_list = []
    x_axis = []
    # vertices_counter: shape(vertices_number,3)
    vertices_counter = tf.ones(mesh_data['vertices'].shape,dtype=tf.int32)

    if config['use_momentum_gradient'] is True:
        accumulation_gradient = tf.zeros(mesh_data['vertices'].shape, dtype=tf.float32)

    vertices_gradient_change_sum = tf.zeros(mesh_data['vertices'].shape,dtype=tf.float32)
    num_times_wrong_classification = 0

    class_weights = tf.Variable(class_weights, trainable=True, dtype=tf.float32)
    cluster_labels = tf.constant(cluster_labels,dtype=tf.int32)
    # 根据每个顶点的类别  以及每个类别的攻击权重，来确定顶点的攻击权重
    # vertex_attack_weights = tf.gather(class_weights, cluster_labels)


    # Defining the attack
    # loss = y_true * log(y_true / y_pred)
    kl_divergence_loss = tf.keras.losses.KLDivergence()
    w = config['attacking_weight']
    if config['dataset'] == 'SHREC11':
        one_hot_original_label_vetor = tf.one_hot(config['source_label'], 30)
    elif config['dataset'] == 'MODELNET40':
        one_hot_original_label_vetor = tf.one_hot(config['source_label'], 40)
    else:
        one_hot_original_label_vetor = config['source_label']

    # Time measurment parameter
    start_time_100_iters = time.time()

    normal_per_face_orig = perceptual_loss.compute_face_normals(mesh_vertices_clean, mesh_faces)
    area_per_vertex_orig, normal_per_vertex_orig = perceptual_loss.get_area_surround_per_vertex_and_normal_per_vertex(mesh_vertices_clean, mesh_faces)
    curvature_per_vertex_orig = perceptual_loss.get_curvature_per_vertex(mesh_vertices_clean, mesh_edges, normal_per_vertex_orig)
    non_weighted_Laplacian_matrix = perceptual_loss.get_non_weighted_Laplacian(mesh_vertices_clean, mesh_faces, mesh_edges, mesh_edge_pairs)
    area_per_face_orig = perceptual_loss.compute_face_areas(mesh_vertices_clean, mesh_faces)

    # curr_perceptual_loss_bound = config['first_bound']
    # curr_adaptive_perceptual_loss_bound = config['first_bound']
    # final_perceptual_loss_bound = config['final_bound']
    # final_adaptive_perceptual_loss_bound = config['final_bound']

    for num_iter in range(config['max_iter']):
        # Extract features and labels,features include (x,y,z,i)
        features, labels = dataset.mesh_data_to_walk_features(mesh_data, network_params)
        ftrs = tf.cast(features[:, :, :3], tf.float32)
        # 这两步其实就是将features拆开成ftrs和v_indices
        # v_indices:组成walk的顶点的索引，长度是walk_len
        v_indices = features[0, :, 3].astype(np.int)  # the vertices indices of the walk
        v_indices_ = features[:, :, 3].astype(np.int)
        # v_indices_expanded = np.expand_dims(v_indices, axis=-1)

        with tf.GradientTape() as tape:
            tape.watch(ftrs)
            pred = network_dnn_model(ftrs, classify=True, training=False)
            # 把network_dnn_model当成函数,输入ftrs,输出pred
            # 获取每个顶点对应的攻击权重
            # v_indices_ = np.expand_dims(v_indices_, axis=-1)
            v_indices_ = tf.constant(v_indices_)

            # Produce the attack
            # 可以把攻击看成是一种损失函数,方向是-loss方向,参数的改变量,即ftrs的改变量
            kld_loss = kl_divergence_loss(one_hot_original_label_vetor, pred)
            attack =  kld_loss
        pred = tf.reduce_sum(pred, 0)
        pred /= network_params.n_walks_per_model

        source_pred_brfore_attack = (pred.numpy())[config['source_label']]
        gradients = tape.gradient(kld_loss, ftrs)
        if config['use_momentum_gradient'] is True:
            # 需要维护一个累计梯度表
            momentum_gradients = tf.gather(accumulation_gradient,v_indices_)
            l1_normalize_gradients, _= tf.linalg.normalize(gradients, 1, (1, 2))
            gradients = config['decay_factor'] * momentum_gradients + l1_normalize_gradients

        mesh_to_save['vertices'] = mesh_vertices.numpy()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(class_weights)
            # 每个顶点的攻击权重
            vertex_attack_weights = tf.gather(class_weights, cluster_labels)
            walk_vertex_attack_weights = tf.gather(vertex_attack_weights, v_indices_)
            walk_vertex_attack_weights = tf.expand_dims(walk_vertex_attack_weights,axis=-1)
            attack_perturbation = w * tf.multiply(gradients, walk_vertex_attack_weights)
            ftrs_after_attack_update = ftrs + attack_perturbation


            # vertices_counter[v_indices] += 1
            updates = tf.ones([v_indices.shape[0],3], dtype=tf.int32)
            v_indices = tf.expand_dims(v_indices, axis=1)
            vertices_counter = tf.tensor_scatter_nd_add(vertices_counter, v_indices, updates)

            # vertices_gradient_change_sum[v_indices] += attack_perturbation[0].numpy()
            # attack_perturbation = tf.squeeze(attack_perturbation,axis=0)
            vertices_gradient_change_sum = tf.tensor_scatter_nd_add(vertices_gradient_change_sum,
                                                                    v_indices,
                                                                    attack_perturbation[0])
            # Updating the mesh itself
            # change = vertices_gradient_change_sum / vertices_counter
            change = tf.math.divide_no_nan(vertices_gradient_change_sum, tf.cast(vertices_counter, dtype=tf.float32))
            mesh_to_save['vertices'] = mesh_vertices.numpy() # todo
            mesh_vertices = mesh_vertices + change

            percept_loss, norm_loss = get_perceptual_loss(loss_weights, mesh_vertices_clean, mesh_vertices,
                                                          mesh_faces, mesh_edges, mesh_edge_pairs,
                                                          area_per_face_orig,
                                                          normal_per_face_orig,
                                                          area_per_vertex_orig,
                                                          normal_per_vertex_orig,
                                                          curvature_per_vertex_orig,
                                                          non_weighted_Laplacian_matrix)

            # adaptive_percept_loss, adaptive_normal_loss = get_GMM_adaptive_perceptual_loss(loss_weights, mesh_vertices_clean, mesh_vertices,
            #                                               mesh_faces, covariances, cluster_labels, mesh_edges, mesh_edge_pairs,
            #                                               area_per_face_orig,
            #                                               normal_per_face_orig,
            #                                               area_per_vertex_orig,
            #                                               normal_per_vertex_orig,
            #                                               curvature_per_vertex_orig,
            #                                               non_weighted_Laplacian_matrix)
            # update vertices after attack
        l2_distance = bound.L2_distance_mesh_adv_origin(mesh_vertices_clean.numpy(), mesh_vertices.numpy())
            
        class_weights_gradients = tape.gradient(percept_loss, class_weights)
        perceptual_optimizer = tf.optimizers.SGD(config['perceptual_attacking_weights_lr'])
        perceptual_optimizer.apply_gradients([(class_weights_gradients, class_weights)])
        del tape

        # print('iter',num_iter,'class_weights_after_opti', class_weights,'reg_grad_max',tf.reduce_max(class_weights_gradients))
        print('iter', num_iter, class_weights)

        threshold_optimizer = tf.optimizers.SGD(config['thre_loss_weight'])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(class_weights)
            w_thre = config['thre_loss_weight']
            thre_loss = threshold_loss(class_weights, class_weights_0, w_thre)

        thre_gradients = tape.gradient(thre_loss, class_weights)
        # print("thre_loss", thre_loss,"thre_grad_max",tf.reduce_max(thre_gradients))
        threshold_optimizer.apply_gradients([(thre_gradients, class_weights)])
        # print('iter',num_iter,'class_weights_after_thed', class_weights)

        if(tf.reduce_any(class_weights < 0)):
            class_weights = tf.Variable(class_weights_0)

        bound_count = 0
        max_need_bound_count = len(config['perceptual_bound']) + len(config['adaptive_perceptual_bound']) + len(config['L2_bound']) + len(config['norm_bound'])
        for curr_perceptual_bound in config['perceptual_bound']:
            curr_need = os.path.join(result_path, "last_model_perceptual_bound_" + str(curr_perceptual_bound) + '.npz')
            if os.path.exists(curr_need) and os.path.isfile(curr_need):
                bound_count += 1
                continue
            if percept_loss > curr_perceptual_bound:
                deform_add_fields_and_dump_model(mesh_data=mesh_to_save, fileds_needed=fields_needed,
                                                 out_fn=curr_need )
        # for curr_adaptive_perceptual_bound in config['adaptive_perceptual_bound']:
        #     curr_need =  os.path.join(result_path, "last_model_adaptive_perceptual_bound_" + str(curr_adaptive_perceptual_bound) + '.npz')
        #     if os.path.exists(curr_need) and os.path.isfile(curr_need):
        #         bound_count += 1
        #         continue
        #     if adaptive_percept_loss > curr_adaptive_perceptual_bound:
        #         deform_add_fields_and_dump_model(mesh_data=mesh_to_save, fileds_needed=fields_needed,
        #                                          out_fn=curr_need)
        for curr_l2_bound in config['L2_bound']:
            curr_need =  os.path.join(result_path, "L2_bound_" + str(curr_l2_bound) + '.npz')
            if os.path.exists(curr_need) and os.path.isfile(curr_need):
                bound_count += 1
                continue
            if l2_distance > curr_l2_bound:
                deform_add_fields_and_dump_model(mesh_data=mesh_to_save, fileds_needed=fields_needed,
                                                 out_fn=curr_need)
        for curr_norm_bound in config['norm_bound']:
            curr_need =  os.path.join(result_path, "norm_bound_" + str(curr_norm_bound) + '.npz')
            if os.path.exists(curr_need) and os.path.isfile(curr_need):
                bound_count += 1
                continue
            if norm_loss > curr_norm_bound:
                deform_add_fields_and_dump_model(mesh_data=mesh_to_save, fileds_needed=fields_needed,
                                                 out_fn=curr_need)
        if bound_count >= max_need_bound_count:
            return

        mesh_to_save['vertices'] = mesh_vertices.numpy()

        # if percept_loss > config['perceptual_bound']:
        #     print("Exiting.. perceptual loss  between clean and adv mesh is too large\n")
        #     break
        mesh_data['vertices'] = mesh_vertices.numpy()
        attack_perturbation_abs = np.abs(attack_perturbation)
        attack_perturbation_sum = np.sum(attack_perturbation_abs)
        max_pert = np.max(attack_perturbation)
        # Check the prediction of the network
        new_pred = network_dnn_model(ftrs_after_attack_update, classify=True, training=False)
        new_pred = tf.reduce_sum(new_pred, 0)
        new_pred /= network_params.n_walks_per_model

        # Check to see that we didn't update too much
        # We don't want the change to be too big, as it may result in intersections.
        # And so, we check to see if the change caused us to get closer to the target by more than 0.01.
        # If so, we will divide the change so it won't change more than 0.01
        source_pred_after_attack = (new_pred.numpy())[config['source_label']]
        source_pred_abs_diff = abs(source_pred_brfore_attack - source_pred_after_attack)
        if source_pred_abs_diff > config['max_label_diff']:
            # We update the gradients accordingly
            ratio = config['max_label_diff'] / source_pred_abs_diff
            gradients = gradients * ratio
        # 更新累计梯度
        if config['use_momentum_gradient'] is True:
            accumulation_gradient = tf.tensor_scatter_nd_update(accumulation_gradient, v_indices, gradients[0])
            # accumulation_gradient = tf.tensor_scatter_nd_add(accumulation_gradient, v_indices, gradients[0])

        # values, indexs = tf.math.top_k(pred, k=2)
        # second_largest_value = tf.cast(values[-1], tf.float32).numpy()
        # second_largest_index = tf.cast(indexs[-1], tf.int32).numpy()
        # # attack info
        # print("iter:", num_iter, " attack:", attack.numpy(), " w:", w, " source prec:",
        #       (pred.numpy())[config['source_label']],
        #       "second prec:", second_largest_value, "second class:", second_largest_index, "max label:",
        #       np.argmax(pred))
        print(" source prec:", (pred.numpy())[config['source_label']], "max label:",np.argmax(pred))
        # print('max_perturbation:', max_pert,'sum_abs_perturbation',attack_perturbation_sum)

        if np.argmax(pred) != config['source_label']:
            num_times_wrong_classification += 1
        else:
            num_times_wrong_classification = 0

        loss.append(attack.numpy())

        # 将更新后的数据回传到mesh_data里，后续的plot和visualization
        # If we got the wrong classification 15 times straight
        if num_times_wrong_classification > 15:
            if num_iter < 15:
                print("\n\nExiting.. Wrong model was loaded / Wrong labels were compared\n\n\n")
                return num_iter
            for curr_perceptual_bound in config['perceptual_bound']:
                curr_need = os.path.join(result_path, "last_model_perceptual_bound_" + str(curr_perceptual_bound) + '.npz')
                if os.path.exists(curr_need) and os.path.isfile(curr_need):
                    continue
                deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                                     out_fn=curr_need)

            # for curr_adaptive_perceptual_bound in config['adaptive_perceptual_bound']:
            #     curr_need = os.path.join(result_path, "last_model_adaptive_perceptual_bound_" + str(
            #         curr_adaptive_perceptual_bound) + '.npz')
            #     if os.path.exists(curr_need) and os.path.isfile(curr_need):
            #         continue
            #     deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
            #                                          out_fn=curr_need)
            for curr_l2_bound in config['L2_bound']:
                curr_need = os.path.join(result_path, "L2_bound_" + str(curr_l2_bound) + '.npz')
                if os.path.exists(curr_need) and os.path.isfile(curr_need):
                    continue
                deform_add_fields_and_dump_model(mesh_data=mesh_to_save, fileds_needed=fields_needed,
                                                     out_fn=curr_need)

            for curr_norm_bound in config['norm_bound']:
                curr_need = os.path.join(result_path, "norm_bound_" + str(curr_norm_bound) + '.npz')
                if os.path.exists(curr_need) and os.path.isfile(curr_need):
                    continue
                deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                                 out_fn=curr_need)
            path = result_path if result_path is not None else None
            whether_misclassified_txt = result_path + '/misclassified.txt'
            last_pred = np.argmax(pred)
            if last_pred != config['source_label']:
                with open(whether_misclassified_txt, 'w') as f:
                    last_pred_class = datasets_labels[last_pred]
                    f.write(f"mesh was misclassified to {last_pred_class} after {num_iter} iterations\n")

            if result_path.__contains__('_meshCNN'):
                deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                                 out_fn=path[0:-4] + '_' + str(id) + 'meshCNN_attacked.npz')
                save_last_class_weight(class_weights.numpy(), result_path + '/last_class_weights.npy')

            else:
                deform_add_fields_and_dump_model(mesh_data=mesh_data, fileds_needed=fields_needed,
                                                 out_fn=path[0:-4] + '_' + str(id) + '_attacked.npz')
                save_last_class_weight(class_weights.numpy(), result_path + '/last_class_weights.npy')
            return num_iter

        # save_image_if_needed()
        # Saving pictures of the models
        if num_iter % 100 == 0:
            total_time_100_iters = time.time() - start_time_100_iters
            start_time_100_iters = time.time()

            preds_to_print_str = ''
            print('\n' + str(net_name) + '\n' + preds_to_print_str + '\n'
                  + 'Time took for 100 iters: ' + str(total_time_100_iters) + '\n')

        curr_save_image_iter = num_iter - (num_iter % config['image_save_iter'])
        if curr_save_image_iter / config['image_save_iter'] >= last_dev_res + 1 or num_iter == 0:
            print(result_path)
            cpos = dump_mesh(mesh_data, result_path, cpos, num_iter, config['x_server_exists'])
            last_dev_res = num_iter / config['image_save_iter']

            deform_add_fields_and_dump_model(mesh_data = mesh_data, fileds_needed = fields_needed,
                                             out_fn=result_path + '/last_model.npz')  # "+ str(num_iter))


        curr_plot_iter = num_iter - (num_iter % config['plot_iter'])
        if curr_plot_iter / config['plot_iter'] >= last_plt_res + 1 or num_iter == 0:
            # plot_preditions(network_params, network_dnn_model, config, mesh_data, result_path, num_iter, x_axis,
            #                 source_pred_list)
            last_plt_res = num_iter / config['plot_iter']

        if config['show_model_every'] > 0 and num_iter % config['show_model_every'] == 0 and num_iter > 0:
            plt.plot(loss)
            plt.show()
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'])

    # transfer mutiple pictures to video
    # result_path = config['result_path']
    # cmd = f'ffmpeg -framerate 24 -i {result_path}/img_%05d.jpg {result_path}/mesh_reconstruction.mp4'
    # os.system(cmd)
    return



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/attack_config/BeayesianMeshWalker/Ours_MeshCNN_attack_bound0.04.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = utils.get_config(opts.config)

    np.random.seed(0)
    utils.config_gpu(-1)
    if config['gpu_to_use'] >= 0:

        utils.set_single_gpu(config['gpu_to_use'])
    config['source_label'] = shrec11_labels.index('alien')
    attack_single_mesh(
        source_mesh="datasets/datasets_processed/shrec16_f500/test/test_alien_T547_not_changed_500.npz",
        config=config, labels=shrec11_labels)

    return 0
if __name__ == '__main__':
    main()
