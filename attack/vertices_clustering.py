from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import numpy as np
from data import dataset
import os
import pandas as pd

def save_means_and_covariances(means, covariances, output_folder):

    np.savetxt(output_folder+'/means.txt', means)
    np.savetxt(output_folder+'/covariances.txt', covariances)
    np.save(output_folder+'/means.npy', means)
    np.save(output_folder+'/covariances.npy', covariances)
    return

def save_cluster_labels(clusters, output_folder):
    np.savetxt(output_folder+'/cluster_labels.txt', clusters)
    np.save(output_folder+'/cluster_labels.npy', clusters)

def vertices_clustering_via_GMM(mesh_data, K, result_path):
    mesh_vertices = mesh_data['vertices']
    mesh_vertex_normals = mesh_data['vertex_normal']
    mesh_faces = mesh_data['faces']
    mesh_edges = mesh_data['edges']
    mesh_vertices_normalized = dataset.norm_model(mesh_vertices)

    mesh_vertices_normalized = mesh_vertices_normalized.astype(np.float64)
    mesh_vertex_normals = mesh_vertex_normals.astype(np.float64)
    mesh_vertex_xyz_dxdydz = np.concatenate((mesh_vertices_normalized, mesh_vertex_normals), axis=1)

    if isinstance(mesh_vertex_xyz_dxdydz, np.ndarray):
        mesh_vertex_df = pd.DataFrame(mesh_vertex_xyz_dxdydz)
    else:
        mesh_vertex_df = mesh_vertex_xyz_dxdydz
    mesh_vertex_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将无穷大值替换为 NaN
    mesh_vertex_df.fillna(0, inplace=True)  # 将 NaN 替换为 0

    # 将清洗后的数据转换回 NumPy 数组（如果需要）
    mesh_vertex_xyz_dxdydz = mesh_vertex_df.values
    # 把mesh_vertices转成变量，因为后续要改变
    # mesh_edge_pairs = tf.convert_to_tensor(mesh_data['edge_pairs'],dtype=tf.int64)

    # mesh_kd_query = mesh_data['kdtree_query']
    # mesh_vertex_xyz_dxdydz = np.concatenate((mesh_vertices, mesh_vertex_normals), axis=1)
    gmm_xyz_dxdydz = GaussianMixture(n_components=K, init_params='kmeans', covariance_type='spherical').fit(mesh_vertex_xyz_dxdydz)  # 将顶点聚类成K类
    # 对每个顶点聚类得到每个顶点的类别
    cluster_labels = gmm_xyz_dxdydz.predict(mesh_vertex_xyz_dxdydz)
    means = gmm_xyz_dxdydz.means_
    covariances = gmm_xyz_dxdydz.covariances_
    clustered_vertices = [[] for _ in range(K)]
    clustered_edges = [[] for _ in range(K)]
    # clustered_kd_query = [[] for _ in range(K)]
    # clustered_face_normals = [[] for _ in range(K)]
    # 初始化一个二维列表，用于存储顶点在mesh里的原始索引
    original_indices = [[] for _ in range(K)]

    # 将相同类别的数据组装到对应的列表中
    for i, label in enumerate(cluster_labels):
        clustered_vertices[label].append(mesh_vertices[i])
        clustered_edges[label].append(mesh_edges[i])
        # clustered_kd_query[label].append(mesh_kd_query[i])
        # clustered_face_normals[label].append(mesh_face_normals[i])
        original_indices[label].append(i)

    # 将聚类后的顶点进行保存，需要时打开
    save_K_obj(mesh_vertices, mesh_faces, cluster_labels,
               output_folder= result_path + '/clustered_vertices/' + '/cluster_to_' + str(K) + '_classes')
    save_means_and_covariances(means, covariances, output_folder= result_path + '/clustered_vertices/' + '/cluster_to_' + str(K) + '_classes')
    save_cluster_labels(cluster_labels, output_folder=result_path + '/clustered_vertices/' + '/cluster_to_' + str(K) + '_classes')
    return means, covariances, cluster_labels

def vertices_clustering_via_KNN(mesh_data, K, result_path):
    mesh_vertices = mesh_data['vertices']
    mesh_vertex_normals = mesh_data['vertex_normal']
    mesh_faces = mesh_data['faces']
    mesh_edges = mesh_data['edges']
    mesh_vertices_normalized = dataset.norm_model(mesh_vertices)
    mesh_vertices_normalized = mesh_vertices_normalized.astype(np.float64)
    mesh_vertex_normals = mesh_vertex_normals.astype(np.float64)
    mesh_vertex_xyz_dxdydz = np.concatenate((mesh_vertices_normalized, mesh_vertex_normals), axis=1)

    # 初始化聚类中心，随机选择k_clusters个点作为初始聚类中心
    centers_indices = np.random.choice(len(mesh_vertex_xyz_dxdydz), K, replace=False)
    centers = mesh_vertex_xyz_dxdydz[centers_indices]
    labels = np.empty(len(mesh_vertex_xyz_dxdydz), dtype=int)
    # 迭代直到聚类中心不再变化或者达到某个停止条件
    while True:
        # 使用KNN找到每个点的最近聚类中心
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(centers)
        distances, indices = nn.kneighbors(mesh_vertex_xyz_dxdydz)

        # 分配聚类标签
        labels[:] = indices.flatten()

        # 更新聚类中心为每个聚类的中心点
        new_centers = np.array([mesh_vertex_xyz_dxdydz[labels == i].mean(axis=0) for i in range(K)])

        # 检查聚类中心是否有显著变化，如果没有则停止迭代
        if np.allclose(centers, new_centers, atol=1e-4):
            break

        centers = new_centers

    cluster_labels = labels
    # 将聚类后的顶点进行保存，需要时打开
    save_K_obj(mesh_vertices, mesh_faces, cluster_labels,
               output_folder= result_path + '/clustered_vertices/' + '/cluster_to_' + str(K) + '_classes')
    # save_means_and_covariances(means, covariances, output_folder= result_path + '/clustered_vertices/' + '/cluster_to_' + str(K) + '_classes')
    save_cluster_labels(cluster_labels, output_folder=result_path + '/clustered_vertices/' + '/cluster_to_' + str(K) + '_classes')
    return cluster_labels


def save_K_obj(vertices, faces, class_labels, output_folder='output'):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 获取类别数量
    num_classes = np.max(class_labels) + 1
    # 遍历每个类别
    for class_label in range(num_classes):
        # 提取属于当前类别的顶点和面
        class_vertices = vertices[class_labels == class_label]
        # 构建输出文件路径
        output_filename = os.path.join(output_folder, f'class_{class_label}.obj')
        # 保存到.obj文件
        save_obj_file(output_filename, class_vertices)

def save_obj_file(filename, vertices):
    with open(filename, 'w') as file:
        for vertex in vertices:
            file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
def main():
    pass
if __name__ == '__main__':
    main()

