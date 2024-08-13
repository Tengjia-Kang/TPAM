import os
import trimesh
import numpy as np
import tensorflow as tf
import copy
from sklearn.mixture import GaussianMixture
# 计算每个对抗样本跟clean之间的L2距离以及  感知约束    曲率difference等
from models.perceptual_loss import get_non_weighted_Laplacian, get_perceptual_loss,compute_face_areas, compute_face_normals, get_GMM_adaptive_perceptual_loss
walker_shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]
walker_shrec11_labels.sort()

def compute_edge_pairs(faces):
    edge_pairs = np.array([[i, j] for face in faces for i in face for j in face if i != j])
    return edge_pairs
def L2_distance_mesh_adv_origin(vertices_clean, vertices_adv):
    '''
      计算adv_mesh 和 clean_mesh 的平均顶点L2距离
      参数:  vertices_clean: ndarray
            vertices_adv: ndarray
    '''
    # 计算对应顶点之间的L2距离
    if isinstance(vertices_adv,np.ndarray) or isinstance(vertices_clean,np.ndarray):
        l2_distances = np.linalg.norm(vertices_adv - vertices_clean, axis=1)
        # 计算平均L2距离
        average_distance = np.mean(l2_distances)
    elif isinstance(vertices_adv,tf.Tensor) or isinstance(vertices_clean,tf.Tensor):
        differ = vertices_adv - vertices_clean
        l2_distances = tf.norm(differ, axis=1)
        average_distance = tf.reduce_mean(l2_distances)
    else:
        raise ValueError("Unsupported data type for vertices_adv and vertices_clean.")
    return average_distance

def get_perceptual_loss_bound(weights, vertices_clean, vertices_adv, faces_clean, faces_adv):

    # # 提取顶点
    # vertices_adv = adv_mesh.vertices
    # vertices_clean = clean_mesh.vertices
    # faces_adv = adv_mesh.faces
    # faces_clean = clean_mesh.faces

    edge_pairs = compute_edge_pairs(np.asarray(faces_clean))
    edges = prepare_edges(vertices_clean, faces_clean)
    # 确保两个网格的顶点数量相同
    if vertices_adv.shape != vertices_clean.shape:
        raise ValueError(
            "The number of vertices in both meshes must be the same for corresponding vertex comparison.")
    non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(
        vertices_clean, faces_clean, edges, edge_pairs)
    edge_pairs = tf.convert_to_tensor(edge_pairs, dtype=tf.int32)
    # convert to tensorflow tensor
    vertices_adv = tf.convert_to_tensor(vertices_adv, dtype=tf.float32)
    vertices_clean = tf.convert_to_tensor(vertices_clean, dtype=tf.float32)
    faces_adv = tf.convert_to_tensor(faces_adv, dtype=tf.int32)
    faces_clean = tf.convert_to_tensor(faces_clean, dtype=tf.int32)

    area_per_face_orig = compute_face_areas(vertices_clean, faces_clean)
    normal_per_face_orig = compute_face_normals(vertices_clean, faces_clean)

    percept_loss = get_perceptual_loss(weights, mesh_orig_vertices=vertices_clean,
                                       mesh_adv_vertices=vertices_adv, faces=faces_adv, edges=None,
                                       edge_pairs=edge_pairs, area_per_face_orig=area_per_face_orig,
                                       normal_per_face_orig=normal_per_face_orig,
                                       area_per_vertex_orig=None, normal_per_vertex_orig=None,
                                       curvature_per_vertex_orig=None,
                                       non_weighted_Laplacian_matrix=non_weighted_Laplacian_matrix)
    return percept_loss


def prepare_edges(vertices, faces):

  edges = [set() for _ in range(vertices.shape[0])] # edge的初始化用集合
  for i in range(faces.shape[0]):
    for v in faces[i]:
      edges[v] |= set(faces[i])

  for i in range(vertices.shape[0]):
    if i in edges[i]:
      edges[i].remove(i)
    edges[i] = list(edges[i])
  max_vertex_degree = np.max([len(e) for e in edges]) # 顶点最大的度，也是edge集合的最大数量，其余用-1填充
  for i in range(vertices.shape[0]):
    if len(edges[i]) < max_vertex_degree:
      edges[i] += [-1] * (max_vertex_degree - len(edges[i]))
  edges = np.array(edges, dtype=np.int32)
  return edges

def save_L2_distance_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path, output_path, dataset_name):
    # find pair mesh

    max_l2_distance = 0  # 初始化最大L2距离为0
    total_l2_distance = 0  # 初始化L2距离的总和
    count_l2_distances = 0  # 初始化L2距离的计算次数
    os.path.dirname(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'a') as f:

        for class_name in walker_shrec11_labels:
            dataset_clean_path_each_class = dataset_clean_path + '/' + class_name + '/' + 'test'
            dataset_adv_path_each_class = dataset_adv_path + '/' + class_name + '/' + 'test'
            if not os.path.exists(dataset_clean_path_each_class) or not os.path.exists(dataset_adv_path_each_class):
                print(f"Paths do not exist: {dataset_clean_path_each_class} or {dataset_adv_path_each_class}")
                continue
            clean_files = os.listdir(dataset_clean_path_each_class)
            adv_files = os.listdir(dataset_adv_path_each_class)
            for adv_file in adv_files:
                if adv_file.endswith('.obj'):
                    adv_id = adv_file.split('_')[1]
                    for clean_file in clean_files:
                        clean_id = clean_file[:-4]
                        if clean_file.endswith('.obj') and adv_id == clean_id:
                            clean_mesh_path = os.path.join(dataset_clean_path_each_class, clean_file)
                            adv_mesh_path = os.path.join(dataset_adv_path_each_class, adv_file)
                            try:
                                # 加载网格
                                adv_mesh = trimesh.load(adv_mesh_path)
                                clean_mesh = trimesh.load(clean_mesh_path)
                                # 提取顶点
                                vertices_adv = adv_mesh.vertices
                                vertices_clean = clean_mesh.vertices

                                # 确保两个网格的顶点数量相同
                                if vertices_adv.shape != vertices_clean.shape:
                                    raise ValueError(
                                        "The number of vertices in both meshes must be the same for corresponding vertex comparison.")

                                # 转换为numpy数组（如果它们还不是）
                                vertices_adv = np.array(vertices_adv)
                                vertices_clean = np.array(vertices_clean)
                                l2_distance = L2_distance_mesh_adv_origin(vertices_clean, vertices_adv)

                                # 写入class_name, clean_id和l2_distance到文件
                                f.write(f"{class_name},{adv_id},{l2_distance}\n")

                                # 更新最大L2距离
                                if l2_distance > max_l2_distance:
                                    max_l2_distance = l2_distance

                                # 累计L2距离和计数
                                total_l2_distance += l2_distance
                                count_l2_distances += 1
                            except Exception as e:
                                print(f"Error calculating L2 distance for {clean_mesh_path} and {adv_mesh_path}: {e}")
            # 计算平均L2距离
        average_l2_distance = total_l2_distance / count_l2_distances if count_l2_distances > 0 else 0

        # 追加最大L2距离和平均L2距离到文件最后
        f.write(f"\nMax L2 Distance: {max_l2_distance}\n")
        f.write(f"Average L2 Distance: {average_l2_distance}\n")
        return average_l2_distance, max_l2_distance

def save_perceptual_loss_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path, output_file,
                                                               dataset_name, weights):
    # find pair mesh
    max_perceptual = 0  # max of datasets
    total_perceptual = 0  # sum of datasets
    count = 0  # count
    # os.path.dirname(output_file)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        for class_name in walker_shrec11_labels:
            dataset_clean_path_each_class = dataset_clean_path + '/' + class_name + '/' + 'test'
            dataset_adv_path_each_class = dataset_adv_path + '/' + class_name + '/' + 'test'
            if not os.path.exists(dataset_clean_path_each_class) or not os.path.exists(dataset_adv_path_each_class):
                print(
                    f"Paths do not exist: {dataset_clean_path_each_class} or {dataset_adv_path_each_class}")
                continue
            clean_files = os.listdir(dataset_clean_path_each_class)
            adv_files = os.listdir(dataset_adv_path_each_class)
            for adv_file in adv_files:
                if adv_file.endswith('.obj'):
                    adv_id = adv_file.split('_')[1]
                    for clean_file in clean_files:
                        clean_id = clean_file[:-4]
                        if clean_file.endswith('.obj') and adv_id == clean_id:
                            clean_mesh_path = os.path.join(dataset_clean_path_each_class, clean_file)
                            adv_mesh_path = os.path.join(dataset_adv_path_each_class, adv_file)
                            try:
                                # 加载网格
                                adv_mesh = trimesh.load(adv_mesh_path)
                                clean_mesh = trimesh.load(clean_mesh_path)
                                # 提取顶点
                                vertices_adv = adv_mesh.vertices
                                vertices_clean = clean_mesh.vertices
                                faces_adv = adv_mesh.faces
                                faces_clean = clean_mesh.faces

                                edge_pairs = compute_edge_pairs(np.asarray(faces_clean))
                                edges = prepare_edges(vertices_clean, faces_clean)
                                # 确保两个网格的顶点数量相同
                                if vertices_adv.shape != vertices_clean.shape:
                                    raise ValueError(
                                        "The number of vertices in both meshes must be the same for corresponding vertex comparison.")
                                non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(
                                    vertices_clean, faces_clean, edges, edge_pairs)
                                edge_pairs = tf.convert_to_tensor(edge_pairs, dtype=tf.int32)
                                # convert to tensorflow tensor
                                vertices_adv = tf.convert_to_tensor(vertices_adv, dtype=tf.float32)
                                vertices_clean = tf.convert_to_tensor(vertices_clean, dtype=tf.float32)
                                faces_adv = tf.convert_to_tensor(faces_adv, dtype=tf.int32)
                                faces_clean = tf.convert_to_tensor(faces_clean, dtype=tf.int32)


                                area_per_face_orig = compute_face_areas(vertices_clean, faces_clean)
                                normal_per_face_orig = compute_face_normals(vertices_clean, faces_clean)
                                percept_loss, norm_loss = get_perceptual_loss(weights, mesh_orig_vertices = vertices_clean,
                                                                   mesh_adv_vertices = vertices_adv, faces = faces_adv, edges = None,
                                                                   edge_pairs = edge_pairs, area_per_face_orig = area_per_face_orig,
                                                                   normal_per_face_orig = normal_per_face_orig ,
                                                                   area_per_vertex_orig = None, normal_per_vertex_orig =None,
                                                                   curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = non_weighted_Laplacian_matrix)

                                # 写入class_name, clean_id和l2_distance到文件
                                f.write(f"{class_name},{adv_id},{percept_loss.numpy()}\n")

                                # 更新最大L2距离
                                if percept_loss > max_perceptual:
                                    max_perceptual = percept_loss
                                # 累计L2距离和计数
                                total_perceptual += percept_loss
                                count += 1
                            except Exception as e:
                                print(f"Error calculating perceptual loss for {clean_mesh_path} and {adv_mesh_path}: {e}")
            # 计算平均L2距离
        average_perceptual = total_perceptual / count if count > 0 else 0
        f.write(f"\nMax perceptual loss: {max_perceptual}\n")
        f.write(f"Average perceptual loss: {average_perceptual}\n")
        return average_perceptual, max_perceptual

def save_norm_loss_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path, output_file,
                                                               dataset_name, weights):
    # find pair mesh
    max_norm_loss = 0  # max of datasets
    total_norm_loss = 0  # sum of datasets
    count = 0  # count
    # os.path.dirname(output_file)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        for class_name in walker_shrec11_labels:
            dataset_clean_path_each_class = dataset_clean_path + '/' + class_name + '/' + 'test'
            dataset_adv_path_each_class = dataset_adv_path + '/' + class_name + '/' + 'test'
            if not os.path.exists(dataset_clean_path_each_class) or not os.path.exists(dataset_adv_path_each_class):
                print(
                    f"Paths do not exist: {dataset_clean_path_each_class} or {dataset_adv_path_each_class}")
                continue
            clean_files = os.listdir(dataset_clean_path_each_class)
            adv_files = os.listdir(dataset_adv_path_each_class)
            for adv_file in adv_files:
                if adv_file.endswith('.obj'):
                    adv_id = adv_file.split('_')[1]
                    for clean_file in clean_files:
                        clean_id = clean_file[:-4]
                        if clean_file.endswith('.obj') and adv_id == clean_id:
                            clean_mesh_path = os.path.join(dataset_clean_path_each_class, clean_file)
                            adv_mesh_path = os.path.join(dataset_adv_path_each_class, adv_file)
                            try:
                                # 加载网格
                                adv_mesh = trimesh.load(adv_mesh_path)
                                clean_mesh = trimesh.load(clean_mesh_path)
                                # 提取顶点
                                vertices_adv = adv_mesh.vertices
                                vertices_clean = clean_mesh.vertices
                                faces_adv = adv_mesh.faces
                                faces_clean = clean_mesh.faces

                                edge_pairs = compute_edge_pairs(np.asarray(faces_clean))
                                edges = prepare_edges(vertices_clean, faces_clean)
                                # 确保两个网格的顶点数量相同
                                if vertices_adv.shape != vertices_clean.shape:
                                    raise ValueError(
                                        "The number of vertices in both meshes must be the same for corresponding vertex comparison.")
                                non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(
                                    vertices_clean, faces_clean, edges, edge_pairs)
                                edge_pairs = tf.convert_to_tensor(edge_pairs, dtype=tf.int32)
                                # convert to tensorflow tensor
                                vertices_adv = tf.convert_to_tensor(vertices_adv, dtype=tf.float32)
                                vertices_clean = tf.convert_to_tensor(vertices_clean, dtype=tf.float32)
                                faces_adv = tf.convert_to_tensor(faces_adv, dtype=tf.int32)
                                faces_clean = tf.convert_to_tensor(faces_clean, dtype=tf.int32)


                                area_per_face_orig = compute_face_areas(vertices_clean, faces_clean)
                                normal_per_face_orig = compute_face_normals(vertices_clean, faces_clean)
                                percept_loss, norm_loss = get_perceptual_loss(weights, mesh_orig_vertices = vertices_clean,
                                                                   mesh_adv_vertices = vertices_adv, faces = faces_adv, edges = None,
                                                                   edge_pairs = edge_pairs, area_per_face_orig = area_per_face_orig,
                                                                   normal_per_face_orig = normal_per_face_orig ,
                                                                   area_per_vertex_orig = None, normal_per_vertex_orig =None,
                                                                   curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = non_weighted_Laplacian_matrix)



                                # 写入class_name, clean_id和l2_distance到文件
                                f.write(f"{class_name},{adv_id},{norm_loss}\n")

                                # 更新最大L2距离
                                if norm_loss > max_norm_loss:
                                    max_norm_loss = norm_loss

                                # 累计L2距离和计数
                                total_norm_loss += norm_loss
                                count += 1

                            except Exception as e:
                                print(f"Error calculating L2 distance for {clean_mesh_path} and {adv_mesh_path}: {e}")
            # 计算平均L2距离
        average_norm_loss = total_norm_loss / count if count > 0 else 0

        # 追加最大L2距离和平均L2距离到文件最后
        f.write(f"\nMax perceptual loss: {max_norm_loss}\n")
        f.write(f"Average perceptual loss: {average_norm_loss}\n")
        return average_norm_loss, max_norm_loss

def norm_model(vertices):
  # Move the model so the bbox center will be at (0, 0, 0)
  mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
  vertices -= mean
  # Scale model to fit into the unit ball
  if 1: # Model Norm -->> !!!
    norm_with = np.max(vertices)
  else:
    norm_with = np.max(np.linalg.norm(vertices, axis=1))
  vertices /= norm_with

  # if norm_model.sub_mean_for_data_augmentation:
  #   vertices -= np.nanmean(vertices, axis=0)
  return vertices
def compute_vertex_normals(vertices, faces):
    num_vertices = len(vertices)
    vertex_normals = np.zeros_like(vertices, dtype=float)
    for face in faces:
        # 获取三角形的三个顶点坐标
        A, B, C = vertices[face]
        # 计算三角形法线
        normal = np.cross(B - A, C - A)
        # 计算三角形面积
        area = 0.5 * np.linalg.norm(normal)
        # 面积加权，累加到每个顶点
        vertex_normals[face] += area * normal
    # 归一化顶点法线
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]
    return vertex_normals
def get_covariances(vertices, faces, K):
    vertices_np = vertices.numpy()
    vertices_np_copy = copy.deepcopy(vertices_np)
    vertices_normalized = norm_model(vertices_np_copy)
    faces_np = faces.numpy()
    vertex_normals = compute_vertex_normals(vertices_np_copy, faces_np)
    mesh_vertex_xyz_dxdydz = np.concatenate((vertices_normalized, vertex_normals), axis=1)
    # mesh_vertex_xyz_dxdydz = np.concatenate((mesh_vertices, mesh_vertex_normals), axis=1)
    gmm_xyz_dxdydz = GaussianMixture(n_components=K, init_params='kmeans', covariance_type='spherical').fit(
        mesh_vertex_xyz_dxdydz)  # 将顶点聚类成K类
    # 对每个顶点聚类得到每个顶点的类别
    cluster_labels = gmm_xyz_dxdydz.predict(mesh_vertex_xyz_dxdydz)
    means = gmm_xyz_dxdydz.means_
    covariances = gmm_xyz_dxdydz.covariances_
    return covariances, cluster_labels


def save_adaptive_perceptual_loss_between_adversaries_and_original_mesh(ROOT_DIR, dataset_clean_path, dataset_adv_path, output_file,
                                                               dataset_name, weights, K =None):
    # find pair mesh
    max_adaptive_perceptual = 0  # max of datasets
    total_adaptive_perceptual = 0  # sum of datasets
    count = 0  # count

    os.path.dirname(output_file)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        for class_name in walker_shrec11_labels:
            dataset_clean_path_each_class = dataset_clean_path + '/' + class_name + '/' + 'test'
            dataset_adv_path_each_class = dataset_adv_path + '/' + class_name + '/' + 'test'
            if not os.path.exists(dataset_clean_path_each_class) or not os.path.exists(dataset_adv_path_each_class):
                print(
                    f"Paths do not exist: {dataset_clean_path_each_class} or {dataset_adv_path_each_class}")
                continue
            clean_files = os.listdir(dataset_clean_path_each_class)
            adv_files = os.listdir(dataset_adv_path_each_class)
            for adv_file in adv_files:
                if adv_file.endswith('.obj'):
                    adv_id = adv_file.split('_')[1]
                    for clean_file in clean_files:
                        clean_id = clean_file[:-4]
                        if clean_file.endswith('.obj') and adv_id == clean_id:
                            clean_mesh_path = os.path.join(dataset_clean_path_each_class, clean_file)
                            adv_mesh_path = os.path.join(dataset_adv_path_each_class, adv_file)
                            category = clean_mesh_path.split('/')[-3]
                            id = clean_mesh_path.split('/')[-1][:-4]
                            curr_folder = category +  '_' + id
                            covariances_path = ROOT_DIR + '/' + curr_folder + '/' + 'clustered_vertices' + '/' + 'cluster_to_'+ str(K) +'_classes' + '/' + 'covariances.npy'
                            cluster_labels_path = ROOT_DIR + '/' + curr_folder + '/' + 'clustered_vertices' + '/' + 'cluster_to_'+ str(K) +'_classes' + '/' + 'cluster_labels.npy'
                            try:
                                # 加载网格
                                adv_mesh = trimesh.load(adv_mesh_path)
                                clean_mesh = trimesh.load(clean_mesh_path)
                                # 提取顶点
                                vertices_adv = adv_mesh.vertices
                                vertices_clean = clean_mesh.vertices
                                faces_adv = adv_mesh.faces
                                faces_clean = clean_mesh.faces

                                edge_pairs = compute_edge_pairs(np.asarray(faces_clean))
                                edges = prepare_edges(vertices_clean, faces_clean)

                                # covariances_path =
                                # 确保两个网格的顶点数量相同
                                if vertices_adv.shape != vertices_clean.shape:
                                    raise ValueError(
                                        "The number of vertices in both meshes must be the same for corresponding vertex comparison.")
                                non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(
                                    vertices_clean, faces_clean, edges, edge_pairs)
                                edge_pairs = tf.convert_to_tensor(edge_pairs, dtype=tf.int32)
                                # convert to tensorflow tensor
                                vertices_adv = tf.convert_to_tensor(vertices_adv, dtype=tf.float32)
                                vertices_clean = tf.convert_to_tensor(vertices_clean, dtype=tf.float32)
                                faces_adv = tf.convert_to_tensor(faces_adv, dtype=tf.int32)
                                faces_clean = tf.convert_to_tensor(faces_clean, dtype=tf.int32)
                                covariances = np.load(covariances_path, allow_pickle=True)
                                cluster_labels = np.load(cluster_labels_path, allow_pickle=True)
                                area_per_face_orig = compute_face_areas(vertices_clean, faces_clean)
                                normal_per_face_orig = compute_face_normals(vertices_clean, faces_clean)
                                adaptive_percept_loss, adaptive_norm_loss = get_GMM_adaptive_perceptual_loss(weights, mesh_orig_vertices = vertices_clean,
                                                                   mesh_adv_vertices = vertices_adv, faces = faces_adv, covariances = covariances,
                                                                   cluster_labels =cluster_labels, edges = None,
                                                                   edge_pairs = edge_pairs, area_per_face_orig = area_per_face_orig,
                                                                   normal_per_face_orig = normal_per_face_orig ,
                                                                   area_per_vertex_orig = None, normal_per_vertex_orig =None,
                                                                   curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = non_weighted_Laplacian_matrix)



                                # 写入class_name, clean_id和l2_distance到文件
                                f.write(f"{class_name},{adv_id},{adaptive_percept_loss.numpy()}\n")

                                # 更新最大L2距离
                                if adaptive_percept_loss > max_adaptive_perceptual:
                                    max_adaptive_perceptual = adaptive_percept_loss

                                # 累计L2距离和计数
                                total_adaptive_perceptual += adaptive_percept_loss
                                count += 1

                            except Exception as e:
                                print(f"Error calculating L2 distance for {clean_mesh_path} and {adv_mesh_path}: {e}")
            # 计算平均L2距离
        average_adaptive_perceptual = total_adaptive_perceptual / count if count > 0 else 0

        # 追加最大L2距离和平均L2距离到文件最后
        f.write(f"\nMax adaptive perceptual loss: {max_adaptive_perceptual}\n")
        f.write(f"Average adaptive perceptual loss: {average_adaptive_perceptual}\n")
        return average_adaptive_perceptual, max_adaptive_perceptual


# L2 threhold
if __name__ == '__main__':
    dataset_name = 'shrec11'
    dataset_clean_path = '/home/kang/SSD/datasets/shrec_16_f500'
    dataset_adv_path = '//attack_results/MeshCNN/RW_MeshCNN_new_version_maxiter_2000/aa_last_npzs/objs'

    # todo 只弄了shrec11数据集上的，还有modelnet40
    weights =  {"W_edge": 1, 'W_normal': 0.1, 'W_laplacian': 10}

    path_union_base = dataset_adv_path.split('/')[-3]
    path_to_save_L2_distances = "evaluate_results" + '/' + "L2_norm" + '/' + path_union_base + '_l2_distances.txt'
    path_to_save_perceptuals = "evaluate_results" + '/' + "perceptual_loss" + '/' + path_union_base + '_perceptuals.txt'

    save_L2_distance_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path,
                                                    path_to_save_L2_distances
                                                           , dataset_name)
    save_perceptual_loss_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path,
                                        path_to_save_perceptuals,
                                                               dataset_name, weights)
