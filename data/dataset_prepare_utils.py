import trimesh
import numpy as np
import re

FIX_BAD_ANNOTATION_HUMAN_15 = 0

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
# manifold_labels = [
#   'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
#   'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
#   'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
#   'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
# ]
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

model_net_shape2label = {v: k for k, v in enumerate(model_net_modelnet40_labels)}

cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}
def extract_class_idx_from_file_path(file, imitating_net):
    if (imitating_net == "MeshCNN"):
        splited_file_with_extension = file.split('/')[-1]
        class_name = file.split('/')[-3]
        mesh_idx = re.sub(r'\D', '', splited_file_with_extension)

    elif (imitating_net == "PDMeshNet"):
        splited_file_with_extension = file.split('/')[-1]
        class_name = file.split('/')[-3]
        mesh_idx = re.sub(r'\D', '', splited_file_with_extension)

    elif (imitating_net == "MeshNet"):
        splited_file_with_extension = file.split('/')[-1]
        class_name = file.split('/')[-3]
        mesh_idx = re.sub(r'\D', '', splited_file_with_extension)

    elif (imitating_net == "ExMeshCNN"):
        splited_file_with_extension = file.split('/')[-1]
        class_name = file.split('/')[-3]
        mesh_idx = re.sub(r'\D', '', splited_file_with_extension)

    elif (imitating_net == "MeshWalker"):
        splited_file_with_extension = file.split('/')[-1]
        class_name = file.split('/')[-3]
        mesh_idx = re.sub(r'\D', '', splited_file_with_extension)

    return class_name, mesh_idx, splited_file_with_extension

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
    if dataset_name.lower().startswith('modelnet') or dataset_name.lower().startswith('manifold'):
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

def calc_vertex_labels_from_face_labels(mesh, face_labels):
  vertices = mesh['vertices']
  faces = mesh['faces']
  all_vetrex_labels = [[] for _ in range(vertices.shape[0])]
  vertex_labels = -np.ones((vertices.shape[0],), dtype=np.int)
  n_classes = int(np.max(face_labels))
  assert np.min(face_labels) == 1  # min label is 1, for compatibility to human_seg labels representation
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


class PredVecDoNotExistError(Exception):
  def __int__(self, message):
    self.message = message
  def __str__(self):
    return self.message

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
    normal = normal / np.linalg.norm(normal)
    # 计算三角形面积
    area = 0.5 * np.linalg.norm(normal)
    # 面积加权，累加到每个顶点
    vertex_normals[face] += area * normal
  # 归一化顶点法线
  vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]
  mesh['vertex_normal'] = vertex_normals
  return vertex_normals