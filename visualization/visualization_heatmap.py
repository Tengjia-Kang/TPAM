#from utils import visualize_npz
import glob, os, json
from easydict import EasyDict
from models import rnn_model
from data import dataset
import tensorflow as tf
import numpy as np
from utils import utils
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh
import open3d as o3d


def compare(pos1, faces1, pos2, faces2):
  n, m = pos1.shape[0], pos2.shape[0]
  tmpx = torch.cat([pos1, pos2], dim=0)
  tmpf = torch.cat([faces1, faces2 + n], dim=0)

  color = torch.zeros([n + m], dtype=pos1.dtype, device=pos1.device)
  color[n:] = (pos1 - pos2).norm(p=2, dim=-1)
  # 合并顶点和面数据以进行可视化
  visualize(tmpx, tmpf, color)
def compare_new_version(pos1, faces1, pos2, faces2):
  n, m = pos1.shape[0], pos2.shape[0]
  tmpx = torch.cat([pos1, pos2], dim=0)
  tmpf = torch.cat([faces1, faces2 + n], dim=0)

  color = torch.zeros([n + m], dtype=pos1.dtype, device=pos1.device)
  color[n:] = (pos1 - pos2).norm(p=2, dim=-1)
  # 合并顶点和面数据以进行可视化
  visualize(pos2, faces2, color[n:])
def visualize(pos, faces, intensity=None):

  mesh = _mesh_graph_object(pos, faces, intensity)
  layout = go.Layout(scene=go.layout.Scene(aspectmode="data"))

  # pio.renderers.default="plotly_mimetype"
  fig = go.Figure(data=[mesh],
                  layout=layout)

  fig.update_layout(
    autosize=True,
    margin=dict(l=20, r=20, t=20, b=20),
    # paper_bgcolor="LightSteelBlue"),
  paper_bgcolor="white")
  fig.show()
  return

def show_walk(model, features, one_walk=False, weights=False, pred_cats=None, pred_val=None, labels=None, save_name=''):
  if weights is not False:
    walks = features[:,:,-1]
    for i, walk in enumerate(walks):
      name = '_rank_{}_weight_{:02d}%'.format(i+1, int(weights[i]*100))
      if labels:
        pred_label=labels[pred_cats[i]]
        pred_score=pred_val[i]
        title='{}: {:2.3f}\n weight: {:2.3f}'.format(pred_label, pred_score, weights[i])
      cur_color= 'cyan' if i < len(walks) //2 else 'magenta'  #'cadetblue'  #label2color[gt]
      rendered = utils.visualize_model(dataset.norm_model(model['vertices'], return_val=True),
                                       model['faces'], walk=[list(walk.astype(np.int32))],
                                       jump_indicator=features[i,:,-2],
                                       show_edges=True,
                                       opacity=0.5,
                                       all_colors=cur_color,
                                       edge_color_a='black',
                                       off_screen=True, save_fn=os.path.join(save_name, name), title=title)
    # TODO: save rendered to file
  else:
    for wi in range(features.shape[0]):
      walk = features[wi, :, -1].astype(np.int)
      jumps = features[wi, :, -2].astype(np.bool)
      utils.visualize_model_walk(model['vertices'], model['faces'], walk, jumps)
      if one_walk:
        break


def load_params(logdir):

  # ================ Loading parameters ============== #
  if not os.path.exists(logdir):
    raise(ValueError, '{} is not a folder'.format(logdir))
  try:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    params.net_input += ['vertex_indices']
    params.batch_size = 1
  except:
    raise(ValueError, 'Could not load params.txt from logdir')
  # ================================================== #
  return params

def load_model(params, model_fn=None):
  # ================ Loading architecture ============== #
  if not model_fn:
    model_fn = glob.glob(params.logdir + '/learned_model2keep__*.keras')
    model_fn.sort()
    model_fn = model_fn[-1]
  if params.net == 'HierTransformer':
    import attention_model
    dnn_model = attention_model.WalkHierTransformer(**params.net_params, params=params,
                                                    model_fn=model_fn, model_must_be_load=True)
  else:
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim - 1,
                                     model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)
  return dnn_model


def predict_and_plot(models, logdir, logdir2=None):
  models.sort()
  params = load_params(logdir)
    # load all npzs in folder of filelist
  test_folder = os.path.dirname(params.datasets2use['test'][0])
  list_per_model = [glob.glob(test_folder + '/*{}*'.format(x)) for x in models]
  npzs = [item for sublist in list_per_model for item in sublist]
  test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, None, mode=params.network_task,
                                                           shuffle_size=0, permute_file_names=False, must_run_on_all=True,
                                                           filenames=npzs)
  dnn_model = load_model(params)
  if logdir2 is not None:
    params_2 = load_params(logdir2)
    dnn_model_2 = load_model(params_2)

  for i, data in enumerate(test_dataset):
    name, ftrs, gt = data
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    ftr2use = ftrs[:, :, :-1].numpy()
    gt = gt.numpy()[0]
    model_fn = name.numpy()[0].decode()
    # forward pass through the model
    if params.cross_walk_attn:
      predictions_, weights, per_walk_predictions_ = [x.numpy() for x in dnn_model(ftr2use, classify='visualize', training=False)]
    else:
      predictions_ = dnn_model(ftr2use, classify=True, training=False).numpy()
    if logdir2 is not None:
      if params_2.cross_walk_attn:
        predictions_2, weights2, per_walk_predictions_2 = [x.numpy() for x in
                                                           dnn_model_2(ftr2use, classify='visualize', training=False)]
      else:
        predictions_2 = dnn_model_2(ftr2use, classify=True, training=False).numpy()
    if params.cross_walk_attn:
      # show only weights of walks where Alon's model failed
      # Showing walks with weighted attention - which walks recieved higher weights
      weights = weights.squeeze()
      if len(weights.shape) > 1:
        weights = np.sum(weights,axis=1)
        weights /= np.sum(weights)
      sorted_weights = np.argsort(weights)[::-1]
      sorted_features = ftrs.numpy()[sorted_weights]
      model = dataset.load_model_from_npz(model_fn)
      print(model_fn)
      print('nv: ', model['vertices'].shape[0])
      per_walk_pred = np.argmax(per_walk_predictions_[sorted_weights], axis=1)
      per_walk_scores = [per_walk_predictions_[i, j] for i,j in zip(sorted_weights, per_walk_pred)]
      # if 'modelnet40' in any(params.datasets2use.values()):
      labels = dataset_preprocess.model_net_labels
      save_dir=os.path.join(params.logdir, 'plots', model_fn.split('/')[-1].split('.')[0])
      show_walk(model, sorted_features, weights=weights[sorted_weights],
                pred_cats=per_walk_pred, pred_val=per_walk_scores, labels=labels,
                save_name=save_dir)
      create_gif_from_preds(save_dir, title=model_fn.split('/')[-1].split('.')[0])
      with open(save_dir + '/pred.txt', 'w') as f:
        f.write('Predicted: {}'.format(labels[np.argmax(predictions_)]))
      # TODO: write prediction_2 scores to see the difference in prediction




def create_gif_from_preds(path, title=''):
  files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.png')]
  if not len(files):
    print('Did not find any .png images in {}'.format(path))
  sorted_indices = np.argsort([int(x.split('_')[-3]) for x in files])
  files = [files[i] for i in sorted_indices]
  from PIL import Image
  ims = [Image.open(x) for x in files]
  ims[0].save(os.path.join(path, '{}_animated.gif'.format(title)), save_all=True, append_images=ims[1:], duration=1000)


def compare_attention():
  attn_csv = '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/False_preds_9250.csv'
  orig_csv = '/home/ran/mesh_walker/runs_compare/0095-23.11.2020..15.31__modelnet/False_preds_9222.csv'
  attn_models = []
  orig_models = []
  with open(attn_csv, 'r') as f:
    for row in f:
      attn_models.append(row.split(',')[0])
  with open(orig_csv, 'r') as f:
    for row in f:
      orig_models.append(row.split(',')[0])

  fixed = [x for x in orig_models if x not in attn_models]
  ruined = [x for x in attn_models if x not in orig_models]


  first10_each_class_fp = [glob.glob('/home/ran/mesh_walker/datasets/modelnet40_walker/test_{}*'.format(c)) for c in dataset_preprocess.model_net_labels]
  first10_each_class = ['_'.join(x.split('_')[3:5]) for y in first10_each_class_fp for x in y[:10] if len(y[0].split('_')) ==8]
  first10_each_class += ['_'.join(x.split('_')[3:6]) for y in first10_each_class_fp for x in y[:10] if len(y[0].split('_')) == 9]

  # predict_and_plot(fixed, '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/')
  # predict_and_plot(orig_models, '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/')
  predict_and_plot(first10_each_class, '/home/ran/mesh_walker/runs_compare/0168-03.12.2020..12.37__modelnet_multiwalk/')
  # predict_and_plot(fixed, '/home/ran/mesh_walker/runs_compare/0095-23.11.2020..15.31__modelnet')
  # attn_corrected = ['bed_0558', 'bookshelf_0633', 'bottle_0416', ]
def _mesh_graph_object(pos, faces, intensity=None, scene="scene", showscale=True):
  cpu = torch.device("cpu")
  if type(pos) != np.ndarray:
    pos = pos.to(cpu).clone().detach().numpy()
  if pos.shape[-1] != 3:
    raise ValueError("Vertices positions must have shape [n,3]")
  if type(faces) != np.ndarray:
    faces = faces.to(cpu).clone().detach().numpy()
  if faces.shape[-1] != 3:
    raise ValueError("Face indices must have shape [m,3]")
  if intensity is None:
    intensity = np.ones([pos.shape[0]])
  elif type(intensity) != np.ndarray:
    intensity = intensity.to(cpu).clone().detach().numpy()

  x, z, y = pos.T
  i, j, k = faces.T

  mesh = go.Mesh3d(x=x, y=y, z=z,
                   color='lightpink',
                   intensity=intensity,
                   opacity=1,
                   # colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                   colorscale=[[0, 'white'], [0.8, 'orange'], [1, 'red']],
                   i=i, j=j, k=k,
                   showscale=showscale,
                   scene=scene,
                   lightposition=dict(x=1, y=-1, z=-0.5)
                   )
  return mesh


def visualize_and_compare(adv_pos, faces, orig_pos, orig_faces, intensity=None):
  orig_mesh = _mesh_graph_object(orig_pos, orig_faces, intensity, "scene")
  adv_mesh = _mesh_graph_object(adv_pos, faces, intensity, "scene2")
  """
  # Superimpose original shape to compare.
  n, m = original_pos.shape[0], pos.shape[0]
  compare_pos = torch.cat([original_pos, pos], dim=0)
  compare_faces = torch.cat([original_faces, faces + n], dim=0)
  compare_color = torch.zeros([n + m], dtype=pos.dtype, device=pos.device)
  compare_color[n:] = (pos - original_pos).norm(p=2, dim=-1)

  mesh_cmp = _mesh_graph_object(compare_pos, compare_faces, compare_color, "scene2", showscale=False)

  """
  fig = make_subplots(rows=1, cols=2,
                      specs=[[{"type": "scene"}, {"type": "scene"}]])

  fig.add_trace(
    orig_mesh,
    row=1, col=1
  )

  fig.add_trace(
    adv_mesh,
    row=1, col=2
  )

  fig.update_layout(
    scene=go.layout.Scene(aspectmode="data"),
    scene2=go.layout.Scene(aspectmode="data"),
    autosize=True,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue"
  )

  fig.show()
  return

def visualize_mesh_subplots(vertices_1, vertices_2, mesh_triangles, title_1='First Mesh', title_2='Second Mesh',
                            save_file_1=None, save_file_2=None):
  T1 = o3d.geometry.TriangleMesh()
  T1.vertices = o3d.utility.Vector3dVector(vertices_1)
  T1.triangles = o3d.utility.Vector3iVector(mesh_triangles)
  T1.compute_vertex_normals()

  T2 = o3d.geometry.TriangleMesh()
  T2.vertices = o3d.utility.Vector3dVector(vertices_2)
  T2.triangles = o3d.utility.Vector3iVector(mesh_triangles)
  T2.compute_vertex_normals()

  vis1 = o3d.visualization.VisualizerWithEditing()
  vis1.create_window(window_name=title_1, width=960, height=540, left=0, top=0)
  vis1.add_geometry(T1)

  vis2 = o3d.visualization.VisualizerWithEditing()
  vis2.create_window(window_name=title_2, width=960, height=540, left=960, top=0)
  vis2.add_geometry(T2)

  while True:
    vis1.update_geometry(T1)
    if not vis1.poll_events():
      break
    vis1.update_renderer()
    if save_file_1 is not None:
      vis1.capture_screen_image(save_file_1)

    vis2.update_geometry(T2)
    if not vis2.poll_events():
      break
    vis2.update_renderer()
    if save_file_2 is not None:
      vis2.capture_screen_image(save_file_2)

  vis1.destroy_window()
  vis2.destroy_window()


if __name__ == '__main__':
  np.random.seed(4)
  # clean_mesh_path = "/home/kang/SSD/datasets/shrec_16_f500/bird1/test/T41.obj"
  clean_mesh_path = "//datasets/datasets_processed/shrec16/test/55_simplified_to_4000.obj"
  # adv_mesh_path = "/home/kang/SSD/Projects/TPAM/attack_results/MeshCNN/new_3termLoss_MeshCNN_Surrogate_RandomWalker_attack/aa_L2_bound_0.02/objs/bird1/test/bird1_T41_L2_bou.obj"
  # adv_mesh_path = "/home/kang/SSD/Projects/TPAM/attack_results/PD-MeshNet/PDMeshNet_Surrogate_RandomWalker/aa_L2_bound_0.007/objs/bird1/test/bird1_T41_L2_boun.obj"
  # adv_mesh_path = "attack_results/MeshCNN/MeshCNN_RW/aa_L2_bound_0.020/objs/gorilla/test/bird1_T66_L2_bou.obj"
  # adv_mesh_path = "attack_results/MeshCNN/MeshCNN_Ours_without/aa_L2_bound_0.02/objs/gorilla/test/gorilla_T471_L2_bou.obj"
  adv_mesh_path = "//attack_results/Visualization/mesh-attack/4000/armadillo_55/L2_bound_0.002.obj"
  # adv_mesh_path1 = "/home/kang/SSD/Projects/TPAM/attack_results/Visualization/RW/4000/armadillo_55/L2_bound_0.002.obj"



  clean_mesh = trimesh.load_mesh(clean_mesh_path)
  clean_mesh_data = {'vertices': clean_mesh.vertices, 'faces': clean_mesh.faces,
                     'n_vertices': clean_mesh.vertices.shape[0]}
  adv_mesh = trimesh.load_mesh(adv_mesh_path)
  adv_mesh_data = {'vertices': adv_mesh.vertices, 'faces': adv_mesh.faces, 'n_vertices': adv_mesh.vertices.shape[0]}
  clean_vertices = torch.from_numpy(clean_mesh_data['vertices'])
  clean_faces = torch.from_numpy(clean_mesh_data['faces'])
  adv_vertices = torch.from_numpy(adv_mesh_data['vertices'])
  adv_faces = torch.from_numpy(adv_mesh_data['faces'])

  # visualize_mesh_subplots(clean_vertices, adv_vertices, adv_faces, title_1='src_1' )

  # visualize_and_compare(clean_vertices, clean_faces, adv_vertices, adv_faces)
  # compare(clean_vertices, clean_faces, adv_vertices, adv_faces)
  # compare_new_version(clean_vertices, clean_faces, clean_vertices, clean_faces)
  compare_new_version(clean_vertices, clean_faces, adv_vertices, adv_faces)