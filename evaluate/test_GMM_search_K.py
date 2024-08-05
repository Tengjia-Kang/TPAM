from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy

if __name__ == '__main__':
    file = "//datasets/datasets_to_attack/shrec16_MeshCNN_to_iter_opti_attack/test/test_pliers_T366_not_changed_500.npz"
    orig_mesh_data = np.load(file, allow_pickle = True)
    # mesh_data 字典型,将orig_mesh_data数据转换
    mesh_data = {k: v for k, v in orig_mesh_data.items()}
    mesh_orig = copy.deepcopy(mesh_data)

    mesh_orig_vertices = tf.convert_to_tensor(mesh_orig['vertices'], dtype=tf.float32)
    # 将输入的数据都转成numpy数组
    mesh_vertices = tf.convert_to_tensor(mesh_data['vertices'], dtype=tf.float32)
    # 把mesh_vertices转成变量，因为后续要改变
    mesh_vertices = tf.Variable(mesh_vertices, trainable=False, dtype=tf.float32)
    mesh_edge_pairs = tf.convert_to_tensor(mesh_data['edge_pairs'], dtype=tf.int64)
    mesh_edges = tf.convert_to_tensor(mesh_data['edges'], dtype=tf.int32)
    mesh_kd_query = mesh_data['kdtree_query']

    mesh_faces = tf.convert_to_tensor(mesh_data['faces'], dtype=tf.int32)
    mesh_face_normals = tf.convert_to_tensor(mesh_data['face_normal'], dtype=tf.float32)

    mesh_vertex_normals = mesh_data['vertex_normal']
    mesh_vertex_xyz_dxdydz = np.concatenate((mesh_vertices, mesh_vertex_normals), axis=1)

    # 设置要测试的K值范围
    k_range = range(2, 250)  # 例如，从2到9测试不同的聚类数
    silhouette_avgs = []  # 用于存储每个K值对应的轮廓系数
    max_silhouette = -1  # 初始化最大轮廓系数为-1
    optimal_k = None  # 初始化最优K值为None
    # 遍历不同的K值
    for k in k_range:
        # 使用高斯混合模型进行聚类
        gmm = GaussianMixture(n_components=k, random_state=42,init_params='kmeans', covariance_type='spherical')
        gmm.fit(mesh_vertex_xyz_dxdydz)
        cluster_labels = gmm.predict(mesh_vertex_xyz_dxdydz)
        # 计算并存储轮廓系数
        silhouette_avg = silhouette_score(mesh_vertex_xyz_dxdydz, cluster_labels)
        silhouette_avgs.append(silhouette_avg)
        print(f"K={k}, Silhouette Score={silhouette_avg}")

        # 检查当前轮廓系数是否为最大值
        if silhouette_avg > max_silhouette:
            max_silhouette = silhouette_avg
            optimal_k = k
        # 绘制K值与轮廓系数的关系图
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_avgs, marker='o')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for different values of K')
    plt.suptitle(f"{file.split('/')[-1]}")
    plt.show()
    save_path = file.split('/')[-1][-4]
    plt.savefig(save_path)
    # 输出轮廓系数最大的K值

    if optimal_k is not None:

        print(f"The optimal number of clusters (K) with the highest silhouette score is: {optimal_k}")

    else:
        print("No optimal K found.")