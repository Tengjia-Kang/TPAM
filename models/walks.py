import os

from easydict import EasyDict
import numpy as np

from src.utils import utils
import tensorflow as tf


def jump_to_closest_unviseted(model_kdtree_query, model_n_vertices, walk, enable_super_jump=True):
  for nbr in model_kdtree_query[walk[-1]]:
    if nbr not in walk:
      return nbr
  if not enable_super_jump:
    return None
  # If not fouind, jump to random node
  node = np.random.randint(model_n_vertices)

  return node


def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]

  backward_steps = 1
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    if len(nodes_to_consider):
      to_add = np.random.choice(nodes_to_consider)
      jump = False
    else:
      if i > backward_steps:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = np.random.randint(n_vertices)
        jump = True
    seq[i] = to_add
    jumps[i] = jump
    visited[to_add] = 1
  return seq, jumps

# def get_seq_random_walk_no_jumps(mesh_extra, f0, seq_len):
#   # 如果 mesh_extra['edges'] 是 NumPy 数组，我们需要将其转换为 CuPy 数组
#   if isinstance(mesh_extra['edges'], np.ndarray):
#     nbrs = cp.array(mesh_extra['edges'])
#   else:  # 如果已经是 CuPy 数组，则直接使用
#     nbrs = mesh_extra['edges']
#   n_vertices = mesh_extra['n_vertices']
#
#   # 使用 CuPy 创建数组
#   seq = cp.zeros((seq_len + 1,), dtype=cp.int32)
#   jumps = cp.zeros((seq_len + 1,), dtype=cp.bool_)
#   visited = cp.zeros(n_vertices, dtype=cp.bool_)
#
#   visited[f0] = True
#   seq[0] = f0
#   jumps[0] = False
#
#   for i in range(1, seq_len + 1):
#     this_nbrs = nbrs[seq[i - 1]]
#     nodes_to_consider = cp.array([n for n in this_nbrs if not visited[n]])
#
#     if len(nodes_to_consider) > 0:
#       to_add = cp.random.choice(nodes_to_consider, size=1)[0]  # 注意这里指定了 size=1 并使用索引 [0]
#       jump = False
#     else:
#       to_add = cp.random.randint(0, n_vertices)
#       jump = True
#
#     seq[i] = to_add
#     jumps[i] = jump
#     visited[to_add] = True
#
#     # 显式地将 CuPy 数组转换回 NumPy 数组
#   seq_np = seq.get()
#   jumps_np = jumps.get()
#
#   return seq_np, jumps_np

def get_seq_random_walk_no_jumps_tensor(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']

  seq = tf.Variable(tf.zeros((seq_len + 1,),dtype = tf.int32))
  jumps = tf.Variable(tf.zeros((seq_len + 1),dtype = tf.bool))
  visited = tf.Variable(tf.zeros(n_vertices + 1, dtype=tf.bool))
  f0 = tf.expand_dims(f0, axis=0)

  # 初始化
  seq = tf.tensor_scatter_nd_update(seq, tf.constant([[0]]) , updates=f0)
  f0 = tf.expand_dims(f0, axis=1)
  visited = tf.tensor_scatter_nd_update(visited, f0 , updates=tf.constant([True]))
  jumps = tf.tensor_scatter_nd_update(jumps, tf.constant([[0]]) , updates=tf.constant([True]))
  backward_steps = 1

  for i in range(1, seq_len + 1):
    current_node = seq[i - 1]
    this_nbrs = tf.gather(nbrs, current_node)
    mask_valid_nbrs = tf.not_equal(this_nbrs, -1)
    this_nbrs_valid = tf.boolean_mask(this_nbrs, mask_valid_nbrs)
    mask_not_visited = tf.logical_not(tf.gather(visited, this_nbrs_valid))
    nodes_to_consider = tf.boolean_mask(this_nbrs_valid, mask_not_visited)
    nodes_to_consider = tf.cast(nodes_to_consider, dtype=tf.int32)
    # Check if there are unvisited neighbors
    n_unvisited = tf.shape(nodes_to_consider)[0]
    if n_unvisited > 0:
      to_add = tf.random.uniform((), minval=0, maxval=n_unvisited, dtype=tf.int32)
      to_add = tf.gather(nodes_to_consider, to_add)
      jump = False
    else:
      if i > backward_steps:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        to_add = tf.random.uniform(shape=[], minval=0, maxval=n_vertices, dtype=tf.int32)
        jump = True

    index = tf.constant([[i]])
    # Update tensors
    seq = tf.tensor_scatter_nd_update(seq, index, [to_add])
    jumps = tf.tensor_scatter_nd_update(jumps, index, updates=tf.constant([jump]))
    visited_index = tf.expand_dims(to_add, axis=0)
    visited_index = tf.expand_dims(visited_index,axis=1)
    visited = tf.tensor_scatter_nd_update(visited, visited_index , updates=tf.constant([True]))

  return seq,jumps

def get_seq_random_walk_no_jumps_tensor_before(mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  # Initialize tensors
  seq = tf.TensorArray(dtype=tf.int32, size=seq_len + 1, dynamic_size=True)
  jumps = tf.TensorArray(dtype=tf.bool, size=seq_len + 1, dynamic_size=True)
  visited = tf.Variable(tf.zeros(n_vertices + 1, dtype=tf.bool))

  # Set initial conditions
  visited = tf.tensor_scatter_nd_update(visited, [[f0]], [True])
  seq = seq.write(0, f0)
  jumps = jumps.write(0, False)   # Assuming the initial step is not a jump
  # Define a loop body for random walk

  def body(i, seq, jumps, visited):
    backward_steps = 1
    current_node = seq.read(i - 1)
    this_nbrs = tf.gather(nbrs, current_node)
    mask_valid_nbrs = tf.not_equal(this_nbrs, -1)
    this_nbrs_valid = tf.boolean_mask(this_nbrs, mask_valid_nbrs)

    mask_not_visited = tf.logical_not(tf.gather(visited, this_nbrs_valid))
    nodes_to_consider = tf.boolean_mask(this_nbrs_valid, mask_not_visited)
    # Ensure indices are of the correct type (int32 or int64)
    nodes_to_consider = tf.cast(nodes_to_consider, dtype=tf.int32)
    # Check if there are unvisited neighbors
    n_unvisited = tf.shape(nodes_to_consider)[0]
    if n_unvisited > 0:
      to_add = tf.random.uniform((), minval=0, maxval=n_unvisited, dtype=tf.int32)
      to_add = tf.gather(nodes_to_consider, to_add)
      jump = False
    else:
      if i > backward_steps:
        to_add = seq.read(i - backward_steps - 1,clear_after_read = False)
        backward_steps += 2
      else:
        to_add = tf.random.uniform(shape=[], minval=0, maxval=n_vertices, dtype=tf.int32)
        jump = True

    # Update tensors
    visited = tf.tensor_scatter_nd_update(visited, [[to_add]], [True])
    seq = seq.write(i, to_add)
    jumps = jumps.write(i, jump)

    return i + 1, seq, jumps, visited
    # Define loop condition
  def cond(i, *args):
    return i < seq_len + 1
  # Run the loop

  _, seq, jumps, _ = tf.while_loop(cond, body, [1, seq, jumps, visited], shape_invariants=[
    tf.TensorShape([]),
    tf.TensorShape(None),  # For seq
    tf.TensorShape(None),  # For jumps
    visited.get_shape()
  ])
  # Convert TensorArrays to tensors
  seq = seq.stack()
  jumps = jumps.stack()
  return seq, jumps






def get_seq_random_walk_using_GMM_vertices_and_no_jumps(mesh_extra, f0 ,seq_len):
  MAX_BACKWARD_ALLOWED = np.inf  # 25 * 2
  nbrs = mesh_extra['edges']
  indices = mesh_extra['indices']
  # n_vertices = max(mesh_extra['indices'])
  n_vertices = mesh_extra['n_vertices']  #需要改进的点
  n_vertices = 2200
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True   # visited 是 seq_len+1,最后一个可能是哨兵？
  visited[f0] = True
  seq[0] = f0
  # jumps[0] = [True]
  jumps[0] = True
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[np.where(indices == seq[i-1])]
    this_nbrs = np.array(this_nbrs).flatten()
    # this_nbrs = nbrs[seq[i - 1]] # 该节点能访问的邻接节点的序列
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    # jump_now = np.random.binomial(1, jump_prob) or (backward_steps > MAX_BACKWARD_ALLOWED)
    # 如果回退次数大于阈值，则jump_now默认为true，或者以0.01的概率全局随机跳跃
    # np.random.binomial(1, jump_prob): 这部分使用 NumPy 提供的 binomial 函数生成一个二项分布的随机数。
    # 这个函数的参数是（n, p），其中 n 是试验次数，p 是成功的概率。在这里，试验次数是 1，成功的概率是 jump_prob
    # if len(nodes_to_consider) and not jump_now:
    if len(nodes_to_consider):
      # 如果有邻接顶点，而且不全局跳跃
      to_add = np.random.choice(nodes_to_consider) #在邻接节点中随机添加一个
      jump = False
      # backward_steps = 1 # 重置后退步数
    else:
      while True:
        # i是要处理的当前顶点，如果当前顶点的序号大于要回退
        if i > backward_steps:
          to_add = seq[i - backward_steps - 1] # 选择回退到之前的顶点 i - backward_steps - 1 添加
          backward_steps += 2
        else: # 要进行全局跳跃了
          backward_steps = 1
          to_add = np.random.choice(indices.flatten())
          jump = True
        if to_add in indices:
          break
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump


  return seq,jumps
def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
  MAX_BACKWARD_ALLOWED = np.inf # 25 * 2
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True   # visited 是 seq_len+1,在选取walk时，舍弃掉第一个seq[1:seq_len+1]
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]] # 该节点能访问的邻接节点的序列
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob) or (backward_steps > MAX_BACKWARD_ALLOWED)
    # 如果回退次数大于阈值，则jump_now默认为true，或者以0.01的概率全局随机跳跃
    # np.random.binomial(1, jump_prob): 这部分使用 NumPy 提供的 binomial 函数生成一个二项分布的随机数。
    # 这个函数的参数是（n, p），其中 n 是试验次数，p 是成功的概率。在这里，试验次数是 1，成功的概率是 jump_prob
    if len(nodes_to_consider) and not jump_now:
      # 如果有邻接顶点，而且不全局跳跃
      to_add = np.random.choice(nodes_to_consider) #在邻接节点中随机添加一个
      jump = False
      backward_steps = 1 # 重置后退步数
    else:
      # i是要处理的当前顶点，如果当前顶点的序号大于要回退
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1] # 选择回退到之前的顶点 i - backward_steps - 1 添加
        backward_steps += 2
      else: # 要进行全局跳跃了
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump

  return seq, jumps


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True

  return seq, jumps

def get_seq_random_walk_local_jumps_tensor(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  for i in range(1, seq_len + 1):
    b = min(0, i - 20)
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      seq[i] = np.random.choice(to_consider)
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True

  return seq, jumps
def get_mesh():
  from dataset_preprocess import prepare_edges_and_kdtree, load_mesh, remesh

  model_fn = os.path.expanduser('~') + '/datasets_processed/human_benchmark_sig_17/sig17_seg_benchmark/meshes/test/shrec/10.off'
  mesh = load_mesh(model_fn)
  mesh, _, _ = remesh(mesh, 4000)
  mesh = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'n_faces_orig': np.asarray(mesh.triangles).shape[0]})
  prepare_edges_and_kdtree(mesh)
  mesh['n_vertices'] = mesh['vertices'].shape[0]
  return mesh


def show_walk_on_mesh(mesh):
  vertices = mesh['vertices']
  f0 = np.random.randint(vertices.shape[0])
  walk, jumps = get_seq_random_walk_no_jumps(mesh, f0, seq_len=400)
  # walk, jumps = get_seq_random_walk_random_global_jumps(mesh, f0, seq_len=100)
  utils.visualize_model_walk_new_version(mesh['vertices'], mesh['faces'], walk, jumps)
  # utils.visualize_model_walk_matplotlib_v(mesh['vertices'], mesh['faces'],walk,jumps)
  # utils.visualize_model_walk_mayavi_v(mesh['vertices'], mesh['faces'],walk,jumps)
  # utils.visualize_model_walk_pyvista_v(mesh['vertices'], mesh['faces'],walk,jumps)
  if 0:
    dxdydz = np.diff(vertices[walk], axis=0)
    for i, title in enumerate(['dx', 'dy', 'dz']):
      plt.subplot(3, 1, i + 1)
      plt.plot(dxdydz[:, i])
      plt.ylabel(title)
    plt.suptitle('Walk features on Human Body')
  # utils.visualize_model(mesh['vertices'], mesh['faces'],
  #                              line_width=2, show_edges=1, edge_color_a='gray',
  #                              show_vertices=False, opacity=1.2,
  #                              point_size=6, all_colors='black',
  #                              walk=walk, edge_colors='red')


if __name__ == '__main__':
  utils.config_gpu(False)
  # mesh = get_mesh()
  np.random.seed()
  # show_walk_on_mesh()

  mesh_data = np.load("/home/kang/SSD/Projects/Random-Walks/datasets/data_to_test/test_gorilla_471_not_changed_9216.npz")
  # show_walk_on_mesh(mesh_data)
  vertices = mesh_data['vertices']
  edges = mesh_data['edges']
  faces = mesh_data['faces']
  mesh_extra = {}
  mesh_extra['n_vertices'] = vertices.shape[0]
  mesh_extra['edges'] = mesh_data['edges']
  mesh_extra['kdtree_query'] = mesh_data['kdtree_query']
  mesh_extra['vertices'] = vertices
  mesh_extra['faces'] = faces

  # f0 =  tf.random.uniform(shape=[],minval=0, maxval=vertices.shape[0], dtype=tf.int32)
  # seq, jumps= get_seq_random_walk_no_jumps_tensor(mesh_extra,f0, 250)
  show_walk_on_mesh(mesh_extra)
