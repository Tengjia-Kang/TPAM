import os
import trimesh
import numpy as np
from evaluate import curvature
import torch
from opt_einsum.backends import torch
import tensorflow as tf
# 计算每个对抗样本跟clean之间的L2距离以及  感知约束    曲率difference等

walker_shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
  ]
walker_shrec11_labels.sort()


# def L2_distance_mesh_adv_origin(vertices_clean, vertices_adv):
#     '''
#       计算adv_mesh 和 clean_mesh 的平均顶点L2距离
#       参数:  vertices_clean: ndarray
#             vertices_adv: ndarray
#     '''
#     # 计算对应顶点之间的L2距离
#     if isinstance(vertices_adv,np.ndarray) or isinstance(vertices_clean,np.ndarray):
#         l2_distances = np.linalg.norm(vertices_adv - vertices_clean, axis=1)
#         # 计算平均L2距离
#         average_distance = np.mean(l2_distances)
#     elif isinstance(vertices_adv,tf.Tensor) or isinstance(vertices_clean,tf.Tensor):
#         differ = vertices_adv - vertices_clean
#         l2_distances = tf.norm(differ, axis=1)
#         average_distance = tf.reduce_mean(l2_distances)
#     else:
#         raise ValueError("Unsupported data type for vertices_adv and vertices_clean.")
#     return average_distance

def save_L2_distance_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path, output_path, dataset_name):
    # find pair mesh

    max_l2_distance = 0  # 初始化最大L2距离为0
    total_l2_distance = 0  # 初始化L2距离的总和
    count_l2_distances = 0  # 初始化L2距离的计算次数
    # os.path.dirname(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as f:

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
                                faces_adv = adv_mesh.faces
                                faces_clean = clean_mesh.faces
                                # 确保两个网格的顶点数量相同
                                if vertices_adv.shape != vertices_clean.shape:
                                    raise ValueError(
                                        "The number of vertices in both meshes must be the same for corresponding vertex comparison.")

                                # 转换为numpy数组（如果它们还不是）
                                vertices_adv = np.array(vertices_adv)
                                vertices_adv = torch.from_numpy(vertices_adv)
                                vertices_clean = np.array(vertices_clean)
                                vertices_clean = torch.from_numpy(vertices_clean)

                                faces_clean = np.array(faces_clean)
                                faces_clean = torch.from_numpy(faces_clean)
                                meancurv_diff_l2 = curvature.meancurvature_diff_l2(vertices_adv, vertices_clean, faces_clean)


                                # 写入class_name, clean_id和l2_distance到文件
                                f.write(f"{class_name},{adv_id},{meancurv_diff_l2}\n")

                                # 更新最大L2距离
                                if meancurv_diff_l2 > max_l2_distance:
                                    max_l2_distance = meancurv_diff_l2

                                # 累计L2距离和计数
                                total_l2_distance += meancurv_diff_l2
                                count_l2_distances += 1

                            except Exception as e:
                                print(f"Error calculating curvature L2 distance for {clean_mesh_path} and {adv_mesh_path}: {e}")
            # 计算平均L2距离
        average_l2_distance = total_l2_distance / count_l2_distances if count_l2_distances > 0 else 0

        # 追加最大L2距离和平均L2距离到文件最后
        f.write(f"\nMax curvature L2: {max_l2_distance}\n")
        f.write(f"Average curvature for datasets: {average_l2_distance}\n")
        return average_l2_distance, max_l2_distance


# L2 threhold
if __name__ == '__main__':
    dataset_name = 'shrec11'
    dataset_clean_path = '/home/kang/SSD/datasets/shrec_16_f500'
    dataset_adv_path = '//attack_results/RW_MeshWalker_new_version_maxiter_3200/aa_last_npzs/objs'
    output_file = "evaluate_results/RW_MeshWalker_new_version_maxiter_3200.txt"
    # todo 只弄了shrec11数据集上的，还有modelnet40
    save_L2_distance_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path, output_file
                                                           , dataset_name)

