import glob, os, shutil
from pathlib import Path
import trimesh
import open3d
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import re
import tensorflow as tf
from utils import utils
import pickle
from data import dataset_prepare_utils
FIX_BAD_ANNOTATION_HUMAN_15 = 0
# Labels for all datasets_processed
# -----------------------
sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}
model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}
cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}
shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_labels.sort()
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}
coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}
manifold_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]

# 跟mesh_net保持一致
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

def calc_mesh_area(mesh):
  t_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=False)
  mesh['area_faces'] = t_mesh.area_faces
  mesh['area_vertices'] = np.zeros((mesh['vertices'].shape[0]))
  for f_index, f in enumerate(mesh['faces']):
    for v in f:
      mesh['area_vertices'][v] += mesh['area_faces'][f_index] / f.size
def get_pred_vec_from_mesh_idx(mesh_idx, triple):
  for tup in triple:
    if tup[0] == mesh_idx:
      return tup[2]
  return
def compute_area_weighted_vertex_normals(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
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
  mesh['vertex_normal'] = vertex_normals

  return vertex_normals
def calc_vertex_labels_from_face_labels(mesh, face_labels):
  vertices = mesh['vertices']
  faces = mesh['faces']
  all_vetrex_labels = [[] for _ in range(vertices.shape[0])]
  vertex_labels = -np.ones((vertices.shape[0],), dtype=np.int)
  n_classes = int(np.max(face_labels))
  assert np.min(face_labels) == 1 # min label is 1, for compatibility to human_seg labels representation
  v_labels_fuzzy = -np.ones((vertices.shape[0], n_classes))
  for i in range(faces.shape[0]):
    label = face_labels[i]
    for f in faces[i]:
      all_vetrex_labels[f].append(label)
  for i in range(vertices.shape[0]):
    counts = np.bincount(all_vetrex_labels[i])
    vertex_labels[i] = np.argmax(counts)
    v_labels_fuzzy[i] = np.zeros((1, n_classes))
    for j in all_vetrex_labels[i]:
      v_labels_fuzzy[i, int(j) - 1] += 1 / len(all_vetrex_labels[i])
  return vertex_labels, v_labels_fuzzy
def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])] # edge的初始化用集合
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']]) # 顶点最大的度，也是edge集合的最大数量，其余用-1填充
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  # 计算每个顶点的 k 最近邻点索引
  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):

    # d: 一个数组，包含了每个查询点到其最近邻点的距离。
    # i_nbrs: 一个数组，包含了每个查询点最近邻点的索引。
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])


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

def compute_edge_pairs(mesh):
  faces = mesh['faces']
  edge_pairs = np.array([[i, j] for face in faces for i in face for j in face if i != j])
  mesh['edge_pairs'] = edge_pairs

  return edge_pairs

def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        dataset_prepare_utils.prepare_edges_and_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)

  return m

def create_tf_example(m):
  pass

def add_fields_and_dump_model_with_MeshCNN_pred(mesh_data, fileds_needed, out_fn, dataset_name, pred_vector, label = None, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = tf.convert_to_tensor(np.zeros((0,)))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = tf.convert_to_tensor(np.zeros((0,)))
      if field == 'kdtree_query' or field == 'edges':
        dataset_prepare_utils.prepare_edges_and_kdtree(m)
      if field == 'face_normal':
        dataset_prepare_utils.prepare_normal_per_face(m)
      if field == 'vertex_normal':
        compute_area_weighted_vertex_normals(m)
      if field == 'edge_pairs':
        dataset_prepare_utils.compute_edge_pairs(m)
      if field == 'pred_vector':
        m[field] = tf.convert_to_tensor(pred_vector)
      if field == 'label':
        m[field] = tf.convert_to_tensor(label)
  if dump_model:
    # 创建一个tf.train.Example对象
    # m['vertices'] = tf.convert_to_tensor(m['vertices'])
    # example = tf.train.Example()
    # example.features.feature['dataset_name'].bytes_list.value.append(m['dataset_name'])
    # example.features.feature['label'].int32_list.value.append(m['label'])
    # example.features.feature['vertices'].float32_list.value.append(m['vertices'])
    # example.features.feature['faces'].int32_list.value.append(m['vertices'])
    #
    # tfrecord_filename = 'data.tfrecord'
    np.savez(out_fn, **m)
  return m
  #mesh_data, fileds_needed, out_fc_full, dataset_name, pred_vector, origin_label, label, train_with_pred_vector
def add_fields_and_dump_model_union(mesh_data, fileds_needed, out_fn, dataset_name, train_with_pred_vector = False,pred_vector = None,origin_label = None,label = -1, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    #   if field not in m.keys():
    if field == 'labels':
      m[field] = np.zeros((0,))
    if field == 'dataset_name':
      m[field] = dataset_name
    if field == 'walk_cache':
      m[field] = np.zeros((0,))
    if field == 'kdtree_query' or field == 'edges':
      dataset_prepare_utils.prepare_edges_and_kdtree(m)
    if field == 'face_normal':
      dataset_prepare_utils.prepare_normal_per_face(m)
    if field == 'vertex_normal':
      dataset_prepare_utils.compute_area_weighted_vertex_normals(m)
    if field == 'edge_pairs':
      dataset_prepare_utils.compute_edge_pairs(m)
    if field == 'pred_vector':
      m[field] = pred_vector
    if field == 'label':
      if train_with_pred_vector:
        m[field] = origin_label
        print(origin_label)
      else:
        m[field] = label
        print(label)
  if dump_model:
    # 创建一个tf.train.Example对象
    # m['vertices'] = tf.convert_to_tensor(m['vertices'])
    # example = tf.train.Example()
    # example.features.feature['dataset_name'].bytes_list.value.append(m['dataset_name'])
    # example.features.feature['label'].int32_list.value.append(m['label'])
    # example.features.feature['vertices'].float32_list.value.append(m['vertices'])
    # example.features.feature['faces'].int32_list.value.append(m['vertices'])
    #
    # tfrecord_filename = 'data.tfrecord'
    np.savez(out_fn, **m)
  return m

def add_fields_and_dump_model_MeshCNN_to_iter_opti(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        dataset_prepare_utils.prepare_edges_and_kdtree(m)
      if field == 'face_normal':
        dataset_prepare_utils.prepare_normal_per_face(m)
      if field == 'vertex_normal':
        compute_area_weighted_vertex_normals(m)
      if field == 'edge_pairs':
        dataset_prepare_utils.compute_edge_pairs(m)
      if field == 'pred_vector':
        pass
        # m[field] = tf.convert_to_tensor(pred_vector)
      if field == 'label':
        pass
        # m[field] = tf.convert_to_tensor(label)
  if dump_model:
    # 创建一个tf.train.Example对象
    # m['vertices'] = tf.convert_to_tensor(m['vertices'])
    # example = tf.train.Example()
    # example.features.feature['dataset_name'].bytes_list.value.append(m['dataset_name'])
    # example.features.feature['label'].int32_list.value.append(m['label'])
    # example.features.feature['vertices'].float32_list.value.append(m['vertices'])
    # example.features.feature['faces'].int32_list.value.append(m['vertices'])

    # tfrecord_filename = 'data.tfrecord'
    np.savez(out_fn, **m)

  return m
def add_fields_and_dump_model_with_MeshCNN_pred_tensor_version(mesh_data, fileds_needed, out_fn, dataset_name, pred_vector, label, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        dataset_prepare_utils.prepare_edges_and_kdtree(m)
      if field == 'face_normal':
        dataset_prepare_utils.prepare_normal_per_face(m)
      if field == 'vertex_normal':
        compute_area_weighted_vertex_normals(m)
      if field == 'edge_pairs':
        dataset_prepare_utils.compute_edge_pairs(m)
      if field == 'pred_vector':
        m[field] = pred_vector
      if field == 'label':
        m[field] = label
  if dump_model:
    np.savez(out_fn, **m)
  return m


def get_sig17_seg_bm_labels(mesh, file, seg_path):
  # Finding the prebest match file name .. :
  in_to_check = file.replace('obj', 'txt')
  in_to_check = in_to_check.replace('off', 'txt')
  in_to_check = in_to_check.replace('_fix_orientation', '')
  if in_to_check.find('MIT_animation') != -1 and in_to_check.split('/')[-1].startswith('mesh_'):
    in_to_check = '/'.join(in_to_check.split('/')[:-2])
    in_to_check = in_to_check.replace('MIT_animation/meshes_', 'mit/mit_')
    in_to_check += '.txt'
  elif in_to_check.find('/scape/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/scape.txt'
  elif in_to_check.find('/faust/') != -1:
    in_to_check = '/'.join(in_to_check.split('/')[:-1])
    in_to_check += '/faust.txt'

  seg_full_fn = []
  for fn in Path(seg_path).rglob('*.txt'):
    tmp = str(fn)
    tmp = tmp.replace('/segs/', '/meshes/')
    tmp = tmp.replace('_full', '')
    tmp = tmp.replace('shrec_', '')
    tmp = tmp.replace('_corrected', '')
    if tmp == in_to_check:
      seg_full_fn.append(str(fn))
  if len(seg_full_fn) == 1:
    seg_full_fn = seg_full_fn[0]
  else:
    print('\nin_to_check', in_to_check)
    print('tmp', tmp)
    raise Exception('!!')
  face_labels = np.loadtxt(seg_full_fn)

  if FIX_BAD_ANNOTATION_HUMAN_15 and file.endswith('test/shrec/15.off'):
    face_center = []
    for f in mesh.faces:
      face_center.append(np.mean(mesh.vertices[f, :], axis=0))
    face_center = np.array(face_center)
    idxs = (face_labels == 6) * (face_center[:, 0] < 0) * (face_center[:, 1] < -0.4)
    face_labels[idxs] = 7
    np.savetxt(seg_full_fn + '.fixed.txt', face_labels.astype(np.int))

  return face_labels


def get_labels(dataset_name, mesh, file, fn2labels_map=None):
  v_labels_fuzzy = np.zeros((0,))
  if dataset_name == 'faust':
    face_labels = np.load('faust_labels/faust_part_segmentation.npy').astype(np.int)
    vertex_labels, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh, face_labels)
    model_label = np.zeros((0,))
    return model_label, vertex_labels, v_labels_fuzzy
  elif dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
    labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
    e_labels = np.loadtxt(labels_fn)
    v_labels = [[] for _ in range(mesh['vertices'].shape[0])]
    faces = mesh['faces']

    fuzzy_labels_fn = '/'.join(file.split('/')[:-2]) + '/sseg/' + file.split('/')[-1].split('.')[-2] + '.seseg'
    seseg_labels = np.loadtxt(fuzzy_labels_fn)
    v_labels_fuzzy = np.zeros((mesh['vertices'].shape[0], seseg_labels.shape[1]))

    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
      faces_edges = []
      for i in range(3):
        cur_edge = (face[i], face[(i + 1) % 3])
        faces_edges.append(cur_edge)
      for idx, edge in enumerate(faces_edges):
        edge = tuple(sorted(list(edge)))
        faces_edges[idx] = edge
        if edge not in edge2key:
          v_labels_fuzzy[edge[0]] += seseg_labels[edges_count]
          v_labels_fuzzy[edge[1]] += seseg_labels[edges_count]

          edge2key[edge] = edges_count
          edges.append(list(edge))
          v_labels[edge[0]].append(e_labels[edges_count])
          v_labels[edge[1]].append(e_labels[edges_count])
          edges_count += 1

    assert np.max(np.sum(v_labels_fuzzy != 0, axis=1)) <= 3, 'Number of non-zero labels must not acceeds 3!'

    vertex_labels = []
    for l in v_labels:
      l2add = np.argmax(np.bincount(l))
      vertex_labels.append(l2add)
    vertex_labels = np.array(vertex_labels)
    model_label = np.zeros((0,))

    return model_label, vertex_labels, v_labels_fuzzy
  else:
    tmp = file.split('/')[-1]
    model_name = '_'.join(tmp.split('_')[:-1])
    if dataset_name.lower().startswith('modelnet'):
      model_label = model_net_shape2label[model_name]
    elif dataset_name.lower().startswith('cubes'):
      model_label = cubes_shape2label[model_name]
    elif dataset_name.lower().startswith('shrec11'):
      model_name = file.split('/')[-3]
      if fn2labels_map is None:
        model_label = shrec11_shape2label[model_name]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    else:
      raise Exception('Cannot find labels for the dataset')
    vertex_labels = np.zeros((0,))
    return model_label, vertex_labels, v_labels_fuzzy

def fix_labels_by_dist(vertices, orig_vertices, labels_orig):
  labels = -np.ones((vertices.shape[0], ))

  for i, vertex in enumerate(vertices):
    d = np.linalg.norm(vertex - orig_vertices, axis=1)
    orig_idx = np.argmin(d)
    labels[i] = labels_orig[orig_idx]

  return labels

def get_faces_belong_to_vertices(vertices, faces):
  faces_belong = []
  for face in faces:
    used = np.any([v in vertices for v in face])
    if used:
      faces_belong.append(face)
  return np.array(faces_belong)


def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
  labels = labels_orig
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    str_to_add = '_simplified_to_' + str(target_n_faces)
    mesh = mesh.remove_unreferenced_vertices()
    if add_labels and labels_orig.size:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
  else:
    mesh = mesh_orig
    str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

  return mesh, labels, str_to_add


def load_meshes(model_fns):
  f_names = glob.glob(model_fns)
  joint_mesh_vertices = []
  joint_mesh_faces = []
  for fn in f_names:
    mesh_ = trimesh.load_mesh(fn)
    vertex_offset = len(joint_mesh_vertices)
    joint_mesh_vertices += mesh_.vertices.tolist()
    faces = mesh_.faces + vertex_offset
    joint_mesh_faces += faces.tolist()

  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(joint_mesh_vertices)
  mesh.triangles = open3d.utility.Vector3iVector(joint_mesh_faces) \
 \
  # mesh.compute_vertex_normals()

  return mesh


def load_mesh(model_fn, classification=True):
  if 1:  # To load and clean up mesh - "remove vertices that share position"
    if classification:
      mesh_ = trimesh.load_mesh(model_fn, process=True)
      mesh_.remove_duplicate_faces()
    else:
      mesh_ = trimesh.load_mesh(model_fn, process=False)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
  else:
    mesh = open3d.io.read_triangle_mesh(model_fn)

  return mesh

def create_tmp_dataset(model_fn, p_out, n_target_faces):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  mesh_orig = load_mesh(model_fn)
  mesh, labels, str_to_add = remesh(mesh_orig, n_target_faces)
  labels = np.zeros((np.asarray(mesh.vertices).shape[0],), dtype=np.int16)
  mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': 0, 'labels': labels})
  out_fn = p_out + '/tmp'
  add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, 'tmp')


#Gloabal variable that holds all the meshCNN and PD MESHNet vertices and faces npz's
#all_mesh_cnn_files = os.listdir(os.path.expanduser('~') + '/mesh_cnn_faces_and_vertices_npz')
def change_faces_and_vertices(mesh_data, file_name: str):
  name = (re.split(pattern=' |/', string=file_name))[-1]
  name = (re.split(pattern=' |\.', string=name))[0]
  # path_to_meshCNN_file = [file for file in all_mesh_cnn_files if str(file).__contains__(name+'_')]
  path_to_meshCNN_file = None # todo
  mesh_cnn_raw_data = np.load(os.path.expanduser('~') + '/mesh_cnn_faces_and_vertices_npz/' + path_to_meshCNN_file[0])
  mesh_cnn_data = {k: v for k, v in mesh_cnn_raw_data.items()}
  mesh_data['vertices'] = mesh_cnn_data['vertices']
  mesh_data['faces'] = mesh_cnn_data['faces']
  mesh_data['label'] = mesh_cnn_data['label']
  return mesh_data


#Gloabal variable that holds all the copycat's npzs
#all_copycat_shrec11_files = os.listdir('datasets_processed/copycat_shrec11/')
def change_to_copycat_walker(mesh_data, file_name: str):
  name = (re.split(pattern=' |/', string=file_name))[-1]
  name = (re.split(pattern=' |\.', string=name))[0]
  # path_to_meshCNN_file = [file for file in all_copycat_shrec11_files if str(file).__contains__(name+'_')]
  path_to_meshCNN_file = None
  mesh_cnn_raw_data = np.load('datasets_processed/copycat_shrec11/' + path_to_meshCNN_file[0], encoding='latin1', allow_pickle=True)
  mesh_cnn_data = {k: v for k, v in mesh_cnn_raw_data.items()}
  mesh_data['label'] = mesh_cnn_data['label']
  return mesh_data

def prepare_directory_from_scratch_union(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                         size_limit=np.inf, fn_prefix='', verbose=True, classification=True,
                                         adversrial_data = None,
                                         mesh_label_predVector_triple = None, imitating_net = None, train_or_attack = None,
                                         train_with_pred_vector = False, need_perceptual_iter_optim = False
                                         ):
  # 根据数据处理的需求来决定最终npz文件中有哪些项
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query', 'label', 'labels', 'dataset_name', 'labels_fuzzy']

  if train_or_attack == "train" and train_with_pred_vector:
    fileds_needed += ['pred_vector']
  # 训练时不需要计算下面的两个法向量
  if train_or_attack == "attack" and need_perceptual_iter_optim:
    fileds_needed += ['face_normal']
    fileds_needed += ['vertex_normal']
    fileds_needed += ['edge_pairs']

  # 创建p_out
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  # filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]

  # tqdm  显示进度条
  for file in tqdm(filenames, disable = 1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]


    # 提取file路径中的mesh的类别和mesh的序号
    class_name, mesh_idx = dataset_prepare_utils.extract_class_idx_from_file_path(file, imitating_net)

    if train_or_attack == "train" and train_with_pred_vector:
      # 然后在triple列表里找到与当前mesh匹配的预测向量和label
      pred_vector = None
      origin_label = -1

      for tup in mesh_label_predVector_triple:
        if (str(mesh_idx) in tup[0] and str(class_name) in tup[0]):
          pred_vector = tup[2]
          origin_label = int(tup[1])
          break
      try:
        if pred_vector is None:
          pred_vector = np.zeros(40)
        if(pred_vector.size == 0) and (origin_label == -1):
          raise dataset_prepare_utils.PredVecDoNotExistError(f"can not find the matched pred vec of{file}")
      except dataset_prepare_utils.PredVecDoNotExistError as e:
        print(e)

    mesh = load_mesh(file, classification=classification)
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      label, labels_orig, v_labels_fuzzy = dataset_prepare_utils.get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))

    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      # m = add_fields_and_dump_model_with_MeshCNN_pred(mesh_data, fileds_needed, out_fc_full, dataset_name, pred_vector, origin_label)

    if train_with_pred_vector:
      m = add_fields_and_dump_model_union(mesh_data, fileds_needed, out_fc_full, dataset_name, train_with_pred_vector = train_with_pred_vector, pred_vector = pred_vector, origin_label = origin_label, label=label)
    else:
      m = add_fields_and_dump_model_union(mesh_data, fileds_needed, out_fc_full, dataset_name, train_with_pred_vector = train_with_pred_vector, label =label)




def prepare_directory_from_scratch_with_pred_vector(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                                    size_limit=np.inf, fn_prefix='', verbose=True, classification=True, adversrial_data = None,
                                                    triple = None, imitating_net = None, train_or_attack = None,
                                                    train_with_pred_vector = False):

  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  fileds_needed += ['labels_fuzzy']
  if train_with_pred_vector:
    fileds_needed += ['pred_vector']
  # 训练时不需要计算下面的两个法向量
  if train_or_attack == "attack":
    fileds_needed += ['face_normal']
    fileds_needed += ['vertex_normal']
    fileds_needed += ['edge_pairs']


  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]

  # tqdm  显示进度条
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    pred_vector = None
    if(imitating_net == "Mesh_CNN"):
      split_file = file.split('/')[-1]
      class_name = file.split('/')[-3]
      mesh_idx = re.sub(r'\D', '', split_file)
      for tup in triple:
        if (tup[0] == mesh_idx):
          pred_vector = tup[2]
          origin_label = int(tup[1])
          break
    elif(imitating_net == "PD_Mesh_Net"):
      split_file = file.split('/')[-1]
      class_name = file.split('/')[-3]
      mesh_idx = re.sub(r'\D', '', split_file)
      for tup in triple:
        find_mesh_index = tup[0].split('/')[-1][1:-6]
        if(find_mesh_index == mesh_idx):
          pred_vector = tup[2]
          origin_label = int(tup[1])
          break
    elif(imitating_net == "Mesh_Net"):
      split_file = file.split('/')[-1]
      class_name = file.split('/')[-3]
      mesh_idx = re.sub(r'\D', '', split_file)
      for tup in triple:
        find_mesh_index = re.sub(r'\D', '', tup[0].split('/')[-1])
        if (find_mesh_index == mesh_idx):
          pred_vector = tup[2]
          origin_label = int(tup[1])
          break

    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None

      # 应该是在这将以后的ground truth label 换成queried prediction label
      label, labels_orig, v_labels_fuzzy = dataset_prepare_utils.get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      m = add_fields_and_dump_model_with_MeshCNN_pred(mesh_data, fileds_needed, out_fc_full, dataset_name, pred_vector, origin_label)

def prepare_directory_with_real_label_for_iterable(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True, adversrial_data = None,
                                                   triple = None):
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  # fileds_needed += ['labels_fuzzy']
  # 训练时不需要计算下面的两个法向量
  fileds_needed += ['face_normal']
  fileds_needed += ['vertex_normal']
  fileds_needed += ['edge_pairs']
  # fileds_needed += ['pred_vector']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]

  # tqdm  显示进度条
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    split_file = file.split('/')[-1]
    class_name = file.split('/')[-3]
    mesh_idx = re.sub(r'\D', '', split_file)
    # for tup in triple:
    #   if(tup[0] == mesh_idx):
    #     pred_vector = tup[2]
    #     origin_label = int(tup[1])
    #     break
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None

      # 应该是在这将以后的ground truth label 换成queried prediction label
      label, labels_orig, v_labels_fuzzy = dataset_prepare_utils.get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      m = add_fields_and_dump_model_MeshCNN_to_iter_opti(mesh_data, fileds_needed, out_fc_full, dataset_name)

# ------------------------------------------------------- #
def prepare_directory_from_scratch_with_pred_vector_tensor_version(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True, adversrial_data = None,
                                                                   triple = None):
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  fileds_needed += ['labels_fuzzy']
  # 训练时不需要计算下面的两个法向量
  fileds_needed += ['face_normal']
  fileds_needed += ['vertex_normal']
  fileds_needed += ['edge_pairs']
  fileds_needed += ['pred_vector']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  # tqdm  显示进度条
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    split_file = file.split('/')[-1]
    class_name = file.split('/')[-3]
    mesh_idx = re.sub(r'\D', '', split_file)
    for tup in triple:
      if(tup[0] == mesh_idx):
        pred_vector = tup[2]
        origin_label = int(tup[1])
        break
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      # 应该是在这将以后的ground truth label 换成queried prediction label
      label, labels_orig, v_labels_fuzzy = dataset_prepare_utils.get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      m = add_fields_and_dump_model_with_MeshCNN_pred_tensor_version(mesh_data, fileds_needed, out_fc_full, dataset_name, pred_vector, origin_label)





def prepare_shrec11(labels2use=shrec11_labels,
                    path_in='datasets/datasets_raw/shrec16/',
                    p_out='datasets/datasets_processed/shrec16__to_attack'):
  dataset_name = 'shrec11'
  n_target_faces = [500]   # n_target_faces = [np.inf]
  if not os.path.isdir(p_out):
    os.mkdir(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test']:
      pin = path_in + name + '/' + part + '/'

      prepare_directory_from_scratch(dataset_name='shrec11', pathname_expansion=pin + '*.obj',
                                     p_out=p_out + '/' + part, n_target_faces=n_target_faces,fn_prefix=part + '_' +name + '_'
                                     )


def match_mesh_and_predict_vector(MeshCNN_shrec16_path_and_class, MeshCNN_shrec16_predict_logits):

  path = []
  mesh_class = []
  for item in MeshCNN_shrec16_path_and_class:
    path.append(item[0])
    mesh_class.append(item[1])

  idx__class_pred_vec = []
  for i,pth in enumerate(path):
    pth = pth.split('/')[-1]
    mesh_idx = re.sub(r'\D','',pth)
    idx__class_pred_vec.append((mesh_idx, mesh_class[i],MeshCNN_shrec16_predict_logits[i]))


  return idx__class_pred_vec
def prepare_shrec11_with_MeshCNN_predict_tensor_version(labels2use=shrec11_labels,
                                                        path_in='datasets/datasets_raw/shrec16',
                                                        p_out='datasets/datasets_processed/shrec16_with_MeshCNN_pred_tensor_version'):
  dataset_name = 'shrec11'
  n_target_faces = [500]
  with open("output_pred_vector/MeshCNN_shrec16_path_and_class.pkl", 'rb') as f:
    MeshCNN_shrec16_path_and_class = pickle.load(f)
  MeshCNN_shrec16_predict_logits = np.load("output_pred_vector/MeshCNN_shrec16_logits.npy")

  idx_class_pred_vec = match_mesh_and_predict_vector(MeshCNN_shrec16_path_and_class, MeshCNN_shrec16_predict_logits)

  # n_target_faces = [np.inf]
  if not os.path.isdir(p_out):
    os.mkdir(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    # for part in ['test', 'train']:
    for part in ['train']:
      pin = path_in + '/' + name + '/' + part + '/'
      prepare_directory_from_scratch_with_pred_vector(dataset_name='shrec11', pathname_expansion=pin + '*.obj',
                                                      p_out=p_out + '/' + part, n_target_faces=n_target_faces,fn_prefix=part + '_' +name + '_',
                                                      triple=idx_class_pred_vec)

def prepare_directory_from_scratch_with_pred_vector_to_attack(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                                              size_limit=np.inf, fn_prefix='', verbose=True, classification=True, adversrial_data = None
                                                              ):
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  fileds_needed += ['labels_fuzzy']
  fileds_needed += ['face_normal']
  fileds_needed += ['vertex_normal']
  fileds_needed += ['edge_pairs']
  fileds_needed += ['pred_vector']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]

  # tqdm  显示进度条
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    split_file = file.split('/')[-1]
    class_name = file.split('/')[-3]
    mesh_idx = re.sub(r'\D', '', split_file)
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None

      # 应该是在这将以后的ground truth label 换成queried prediction label
      label, labels_orig, v_labels_fuzzy = dataset_prepare_utils.get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0,))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict(
        {'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      # m = add_fields_and_dump_model_with_MeshCNN_pred(mesh_data, fileds_needed, out_fc_full, dataset_name,
      #                                                 origin_label)

def prepare_shrec11_with_MeshCNN_predict_to_attack(labels2use=shrec11_labels,
                                                   path_in='datasets/datasets_raw/shrec16_MeshCNN',
                                                   p_out='datasets/datasets_to_attack/shrec16_with_MeshCNN_to_attack'):
  dataset_name = 'shrec11'
  n_target_faces = [500]
  # with open("output_pred_vector/MeshCNN_shrec16_path_and_class.pkl", 'rb') as f:
  #   MeshCNN_shrec16_path_and_class = pickle.load(f)
  # MeshCNN_shrec16_predict_logits = np.load("output_pred_vector/MeshCNN_shrec16_logits.npy")
  # idx_class_pred_vec = match_mesh_and_predict_vector(MeshCNN_shrec16_path_and_class, MeshCNN_shrec16_predict_logits)

  # n_target_faces = [np.inf]
  if not os.path.isdir(p_out):
    os.mkdir(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    # for part in ['test', 'train']:
    for part in ['test']:
      pin = path_in + '/' + name + '/' + part + '/'
      prepare_directory_with_real_label_for_iterable(dataset_name='shrec11', pathname_expansion=pin + '*.obj',
                                                     p_out=p_out + '/' + part, n_target_faces=n_target_faces,
                                                     fn_prefix=part + '_' + name + '_'
                                                     )




def prepare_shrec11_with_MeshCNN_predict(labels2use=shrec11_labels,
                                         path_in='datasets/datasets_raw/shrec16',
                                         p_out='datasets/datasets_processed/shrec16_with_MeshCNN_pred_to_attack'):
  dataset_name = 'shrec11'
  n_target_faces = [500]
  with open("../output_pred_vector/MeshCNN_shrec16_path_and_class.pkl", 'rb') as f:
    MeshCNN_shrec16_path_and_class = pickle.load(f)
  MeshCNN_shrec16_predict_logits = np.load("../output_pred_vector/MeshCNN_shrec16_logits.npy")
  idx_class_pred_vec = match_mesh_and_predict_vector(MeshCNN_shrec16_path_and_class, MeshCNN_shrec16_predict_logits)

  # n_target_faces = [np.inf]
  if not os.path.isdir(p_out):
    os.mkdir(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    # for part in ['test', 'train']:
    for part in ['test', 'train']:
      pin = path_in + '/' + name + '/' + part + '/'
      prepare_directory_from_scratch_with_pred_vector(dataset_name='shrec11', pathname_expansion=pin + '*.obj',
                                                      p_out = p_out + '/' + part, n_target_faces=n_target_faces,fn_prefix=part + '_' +name + '_',
                                                      triple = idx_class_pred_vec,imitating_net="Mesh_CNN")

def prepare_shrec11_with_PD_MeshNet_predict(labels2use=shrec11_labels,
                                            path_in='datasets/datasets_raw/shrec16',
                                            p_out='datasets/datasets_processed/shrec16_with_PDMeshNet_pred_to_train_imitaingNet'):
  dataset_name = 'shrec11'
  n_target_faces = [500]

  PD_MeshNet_path_label_logits = np.load("output_pred_vector/PD_MeshNet_path_label_logits.npy",allow_pickle=True)
  # idx_class_pred_vec = match_mesh_and_predict_vector(MeshCNN_shrec16_path_and_class, MeshCNN_shrec16_predict_logits)
  tripe_list = []
  for tripe in PD_MeshNet_path_label_logits:
    tripe_tuple = tuple(tripe)
    tripe_list.append(tripe_tuple)
  sorted_tripe_list = sorted(tripe_list, key = lambda label: label[1])


  if not os.path.isdir(p_out):
    os.mkdir(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    # for part in ['test', 'train']:
    for part in ['train']:
      pin = path_in + '/' + name + '/' + part + '/'
      prepare_directory_from_scratch_with_pred_vector(dataset_name='shrec11', pathname_expansion=pin + '*.obj',
                                                      p_out = p_out + '/' + part, n_target_faces=n_target_faces,fn_prefix=part + '_' +name + '_',
                                                      triple = sorted_tripe_list,imitating_net = "PD_Mesh_Net")

def prepare_Manifold40_with_MeshNet_predict(labels2use= model_net_modelnet40_labels,
                                            path_in='/home/kang/SSD/datasets/datasets_raw/Manifold40',
                                            p_out='../datasets/datasets_processed/manifold40_with_MeshNet_pred_to_train_imitaingNet'):
  dataset_name = 'manifold40'
  n_target_faces = [500]
  # 先把查询到的预测向量加载进来
  MeshNet_path_label = np.load("../output_pred_vector/MeshNet_path_label.npy",allow_pickle=True)
  MeshNet_logits = np.load("../output_pred_vector/MeshNet_logits.npy",allow_pickle=True)
  # idx_class_pred_vec = match_mesh_and_predict_vector(MeshCNN_shrec16_path_and_class, MeshCNN_shrec16_predict_logits)
  tripe_list = []
  unequal_count = 0
  for i, pair in enumerate(MeshNet_path_label):
    if(np.argmax(MeshNet_logits[i]) != int(pair[1])):
      print("warning, 预测向量最大值不等于真实标签")
      unequal_count += 1
    tripe_tuple = tuple([pair[0],int(pair[1]),MeshNet_logits[i]])
    tripe_list.append(tripe_tuple)



  print(f"有{unequal_count}个输入数据的预测向量最大值不等于真实标签")
  # sorted_tripe_list = sorted(tripe_list, key = lambda label: label[1])

  if not os.path.isdir(p_out):
    os.mkdir(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    # for part in ['test', 'train']:
    for part in ['train']:
      pin = path_in + '/' + name + '/' + part + '/'
      # prepare_directory_from_scratch_with_pred_vector(dataset_name='manifold40', pathname_expansion=pin + '*.obj',
      #                                                 p_out = p_out + '/' + part, n_target_faces=n_target_faces,fn_prefix=part + '_' +name + '_',
      #                                                 triple = tripe_list, imitating_net = "Mesh_Net", train_or_attack="train", train_with_pred_vector = True)
      prepare_directory_from_scratch_union(dataset_name='manifold40', pathname_expansion=pin + '*.obj',
                                           p_out=p_out + '/' + part, n_target_faces=n_target_faces,
                                           fn_prefix=part + '_' + name + '_',
                                           mesh_label_predVector_triple = tripe_list,
                                           imitating_net="Mesh_Net",
                                           train_or_attack="train", train_with_pred_vector=True,
                                           need_perceptual_iter_optim=False,add_labels=True)

    for part in ['test']:
      pin = path_in + '/' + name + '/' + part + '/'
      prepare_directory_from_scratch_union(dataset_name='manifold40', pathname_expansion=pin + '*.obj',
                                           p_out = p_out + '/' + part, n_target_faces=n_target_faces,fn_prefix=part + '_' +name + '_',
                                           mesh_label_predVector_triple=tripe_list,
                                           imitating_net="Mesh_Net",
                                           train_or_attack="train", train_with_pred_vector=False,
                                           need_perceptual_iter_optim=False, add_labels=True)

def prepare_directory_from_scratch(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True, adversrial_data = None):
  fileds_needed = ['vertices', 'faces', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  fileds_needed += ['labels_fuzzy']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      label, labels_orig, v_labels_fuzzy = dataset_prepare_utils.get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      if adversrial_data == 'mesh_CNN' or adversrial_data == 'PD_MeshNet':
        change_faces_and_vertices(mesh_data, str(file))
      if adversrial_data == 'walker_copycat':
        change_to_copycat_walker(mesh_data, str(file))
      m = add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)

def calc_face_labels_after_remesh(mesh_orig, mesh, face_labels):
  t_mesh = trimesh.Trimesh(vertices=np.array(mesh_orig.vertices), faces=np.array(mesh_orig.triangles), process=False)

  remeshed_face_labels = []
  for face in mesh.triangles:
    vertices = np.array(mesh.vertices)[face]
    center = np.mean(vertices, axis=0)
    p, d, closest_face = trimesh.proximity.closest_point(t_mesh, [center])
    remeshed_face_labels.append(face_labels[closest_face[0]])
  return remeshed_face_labels



def prepare_human_body_segmentation():
  dataset_name = 'sig17_seg_benchmark'
  labels_fuzzy = True
  human_seg_path = os.path.expanduser('~') + '/mesh_walker/datasets_raw/sig17_seg_benchmark/'
  p_out = os.path.expanduser('~') + '/mesh_walker/datasets_processed-tmp/sig17_seg_benchmark-no_simplification/'

  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name', 'face_labels']
  if labels_fuzzy:
    fileds_needed += ['labels_fuzzy']

  n_target_faces = [np.inf]
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  for part in ['test', 'train']:
    print('part: ', part)
    path_meshes = human_seg_path + '/meshes/' + part
    seg_path = human_seg_path + '/segs/' + part
    all_fns = []
    for fn in Path(path_meshes).rglob('*.*'):
      all_fns.append(fn)
    for fn in tqdm(all_fns):
      model_name = str(fn)
      if model_name.endswith('.obj') or model_name.endswith('.off') or model_name.endswith('.ply'):
        new_fn = model_name[model_name.find(part) + len(part) + 1:]
        new_fn = new_fn.replace('/', '_')
        new_fn = new_fn.split('.')[-2]
        out_fn = p_out + '/' + part + '__' + new_fn
        mesh = mesh_orig = load_mesh(model_name, classification=False)
        mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
        face_labels = get_sig17_seg_bm_labels(mesh_data, model_name, seg_path)
        labels_orig, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh_data, face_labels)
        if 0: # Show segment borders
          b_vertices = np.where(np.sum(v_labels_fuzzy != 0, axis=1) > 1)[0]
          vertex_colors = np.zeros((mesh_data['vertices'].shape[0],), dtype=np.int)
          vertex_colors[b_vertices] = 1
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=vertex_colors, point_size=2)
        if 0: # Show face labels
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], face_colors=face_labels, show_vertices=False, show_edges=False)
        if 0:
          print(model_name)
          print('min: ', np.min(mesh_data['vertices'], axis=0))
          print('max: ', np.max(mesh_data['vertices'], axis=0))
          cpos = [(-3.5, -0.12, 6.0), (0., 0., 0.1), (0., 1., 0.)]
          utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=labels_orig, cpos=cpos)
        add_labels = 1
        label = -1
        for this_target_n_faces in n_target_faces:
          mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
          if mesh == mesh_orig:
            remeshed_face_labels = face_labels
          else:
            remeshed_face_labels = calc_face_labels_after_remesh(mesh_orig, mesh, face_labels)
          mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices),
                                'faces': np.asarray(mesh.triangles),
                                'label': label, 'labels': labels,
                                'face_labels': remeshed_face_labels})
          if 1:
            v_labels, v_labels_fuzzy = calc_vertex_labels_from_face_labels(mesh_data, remeshed_face_labels)
            mesh_data['labels'] = v_labels
            mesh_data['labels_fuzzy'] = v_labels_fuzzy
          if 0:  # Show segment borders
            b_vertices = np.where(np.sum(v_labels_fuzzy != 0, axis=1) > 1)[0]
            vertex_colors = np.zeros((mesh_data['vertices'].shape[0],), dtype=np.int)
            vertex_colors[b_vertices] = 1
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=vertex_colors, point_size=10)
          if 0:  # Show face labels
            utils.visualize_model(np.array(mesh.vertices), np.array(mesh.triangles), face_colors=remeshed_face_labels, show_vertices=False, show_edges=False)
          out_fc_full = out_fn + str_to_add
          if os.path.isfile(out_fc_full + '.npz'):
            continue
          add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)
          if 0:
            utils.visualize_model(mesh_data['vertices'], mesh_data['faces'], vertex_colors_idx=mesh_data['labels'].astype(np.int),
                                  cpos=[(-2., -0.2, 3.3), (0., -0.3, 0.1), (0., 1., 0.)])





def map_fns_to_label(path=None, filenames=None):
  lmap = {}
  if path is not None:
    iterate = glob.glob(path + '/*.npz')
  elif filenames is not None:
    iterate = filenames

  for fn in iterate:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    label = int(mesh_data['label'])
    if label not in lmap.keys():
      lmap[label] = []
    if path is None:
      lmap[label].append(fn)
    else:
      lmap[label].append(fn.split('/')[-1])
  return lmap


def change_train_test_split(path, n_train_examples, n_test_examples, split_name):
  np.random.seed()
  fns_lbls_map = map_fns_to_label(path)
  for label, fns_ in fns_lbls_map.items():
    fns = np.random.permutation(fns_)
    assert len(fns) == n_train_examples + n_test_examples
    train_path = path + '/' + split_name + '/train'
    if not os.path.isdir(train_path):
      os.makedirs(train_path)
    test_path = path + '/' + split_name + '/test'
    if not os.path.isdir(test_path):
      os.makedirs(test_path)
    for i, fn in enumerate(fns):
      out_fn = fn.replace('train_', '').replace('test_', '')
      if i < n_train_examples:
        shutil.copy(path + '/' + fn, train_path + '/' + out_fn)
      else:
        shutil.copy(path + '/' + fn, test_path + '/' + out_fn)


# ------------------------------------------------------- #


def prepare_one_dataset(dataset_name, mode):
  dataset_name = dataset_name.lower()
  if dataset_name == 'modelnet40' or dataset_name == 'modelnet':
    pass
    # prepare_modelnet40()

  if dataset_name == 'shrec11':
    pass

  if dataset_name == 'cubes':
    pass

def vertex_pertubation(faces, vertices):
  n_vertices2change = int(vertices.shape[0] * 0.3)
  for _ in range(n_vertices2change):
    face = faces[np.random.randint(faces.shape[0])]
    vertices_mean = np.mean(vertices[face, :], axis=0)
    v = np.random.choice(face)
    vertices[v] = vertices_mean
  return vertices


def visualize_dataset(pathname_expansion):
  cpos = None
  filenames = glob.glob(pathname_expansion)
  while 1:
    fn = np.random.choice(filenames)
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    vertex_colors_idx = mesh_data['labels'].astype(np.int) if mesh_data['labels'].size else None
    vertices = mesh_data['vertices']
    #vertices = vertex_pertubation(mesh_data['faces'], vertices)
    utils.visualize_model(vertices, mesh_data['faces'], vertex_colors_idx=vertex_colors_idx, cpos=cpos, point_size=5)

if __name__ == '__main__':
  TEST_FAST = 0
  utils.config_gpu(True)
  np.random.seed(1)
  # prepare_shrec11()
  # prepare_shrec11_with_MeshWalker_predict()
  # prepare_shrec11_with_PD_MeshNet_predict()
  # prepare_shrec11_with_MeshCNN_predict()
  # prepare_Manifold40_with_MeshNet_predict()
  prepare_Manifold40_with_MeshNet_predict()
  # prepare_shrec11_with_MeshCNN_predict_to_attack()
  # prepare_shrec11_with_MeshCNN_predict()
