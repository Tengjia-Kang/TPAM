import numpy as np
import tensorflow as tf
import trimesh
import time

def compute_face_normals(vertices, faces):
    face_vertices = tf.gather(vertices, faces)
    # 计算面的两个边向量
    edge1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    edge2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
    # 计算面的法向量
    face_normals = tf.linalg.cross(edge1, edge2)
    # 归一化法向量
    normalized_face_normals = tf.nn.l2_normalize(face_normals, axis=-1)
    return normalized_face_normals



def compute_face_areas(vertices, faces):
    # 提取面的顶点坐标
    face_vertices = tf.gather(vertices, faces)
    # 计算两个边的向量
    edge1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    edge2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
    # 计算叉积得到法线向量
    face_normals = tf.linalg.cross(edge1, edge2)
    # 计算法线向量的模长，即三角形面积的两倍
    face_area_twice = tf.norm(face_normals, axis=1)
    # 除以2得到三角形面积
    face_areas = face_area_twice / 2.0
    return face_areas

# def get_normal_per_vertex(vertices, faces, area_per_face):
#     # compute area-weighted vertex normal
#     num_vertices = tf.shape(vertices)[0]
#
#     # normal_per_vertex_array = tf.TensorArray(dtype=tf.float32, size=num_vertices)
#
#     for i in range(num_vertices):
#         # find all triangle include i-th vertex
#         i_faces_indices = tf.where(tf.reduce_any(tf.equal(faces, i), axis=1))
#         # squeeze dim
#         i_faces_indices = tf.squeeze(i_faces_indices)
#         if tf.shape(i_faces_indices)[0] == 0:
#             continue
#         i_faces = tf.gather(faces, i_faces_indices)
#
#         if tf.shape(i_faces)[0] < 3:
#             continue
#         # then we got all triangle vertex include i-vertex
#         # a tensor shape (N, 3, 3)
#         # N represents number of triangle include i-vertex
#         # first 3 represents three points, last 3 represents (x,y,z)
#         i_faces_vertices = tf.gather(vertices, i_faces)
#
#         face_normals = tf.linalg.cross(i_faces_vertices[:, 1, :] - i_faces_vertices[:, 0, :],
#                                          i_faces_vertices[:, 2, :] - i_faces_vertices[:, 0, :])
#
#         face_areas = tf.gather(area_per_face, i_faces_indices)
#         # # 计算每个面片的面积
#         # face_areas = 0.5 * tf.norm(normal_vectors, axis=1)

def get_area_loss(w, area_orig, area_adv, faces):
    # w = weights['W_area']
    n = faces.shape[0]
    area_loss = tf.nn.l2_loss(area_adv - area_orig) / n * 2
    return w * area_loss

def get_area_per_vertex_loss(weights, area_per_vertex_orig, area_per_vertex_adv):
    w = weights['W_area']
    n = area_per_vertex_orig.shape[0]
    area_per_vertex_loss = tf.nn.l2_loss(area_per_vertex_adv, area_per_vertex_orig) / n * 2
    return w * area_per_vertex_loss
def get_adaptive_face_normal_loss(w, normal_per_face_orig, normal_per_face_adv, vertex_coefficient, faces):
    # w = weights['W_normal']
    face_coefficient = tf.gather(vertex_coefficient, faces)
    face_coefficient = tf.reduce_mean(face_coefficient, axis=-1, keepdims=False)
    n = normal_per_face_orig.shape[0]
    # norm_loss = tf.nn.l2_loss(normal_per_face_adv - normal_per_face_orig) * 2 / n
    face_normal_diff = normal_per_face_adv - normal_per_face_orig
    face_normal_diff_suqared = tf.square(face_normal_diff)
    face_coefficient_expanded = tf.expand_dims(face_coefficient, axis = -1)
    face_coefficient_expanded = tf.tile(face_coefficient_expanded, [1, 3])
    face_coefficient_weighted_face_diff_squared = face_normal_diff_suqared * face_coefficient_expanded
    adaptive_norm_loss = tf.reduce_sum(face_coefficient_weighted_face_diff_squared) / n
    return adaptive_norm_loss * w

def covariances_normalize(covariances):
    sum = tf.reduce_sum(covariances)
    covariances_normalized = covariances / sum
    covariances_normalized = tf.cast(covariances_normalized, tf.float32)
    return covariances_normalized
def get_GMM_adaptive_perceptual_loss(weights, mesh_orig_vertices, mesh_adv_vertices, faces, covariances, cluster_labels,
                                     edges=None, edge_pairs =None
                            ,area_per_face_orig = None, normal_per_face_orig = None,
                            area_per_vertex_orig = None, normal_per_vertex_orig =None,
                            curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = None):


    covariances = covariances_normalize(covariances)
    coefficient = 1 - covariances
    vertex_coefficient = tf.gather(coefficient, cluster_labels)
    adaptive_laplacian_loss = get_adaptive_laplacian_regularization_loss(weights['W_laplacian'], mesh_orig_vertices, mesh_adv_vertices,
                                                       non_weighted_Laplacian_matrix, vertex_coefficient)

    normal_per_face_adv = compute_face_normals(mesh_adv_vertices, faces)
    adaptive_face_norm_loss = get_adaptive_face_normal_loss(weights['W_normal'], normal_per_face_orig, normal_per_face_adv,vertex_coefficient, faces)
    adaptive_edge_loss = get_adaptive_edge_loss(weights["W_edge"],mesh_orig_vertices, mesh_adv_vertices,vertex_coefficient, faces, edge_pairs)

    # print(f"adaptive Norm loss: {adaptive_face_norm_loss}")
    adaptive_perceptual_loss = adaptive_edge_loss + adaptive_laplacian_loss + adaptive_face_norm_loss
    # adaptive_perceptual_loss = adaptive_face_norm_loss
    return adaptive_perceptual_loss, adaptive_face_norm_loss

def get_perceptual_loss(weights, mesh_orig_vertices, mesh_adv_vertices, faces, edges=None, edge_pairs =None
                            , area_per_face_orig = None, normal_per_face_orig = None,
                            area_per_vertex_orig = None, normal_per_vertex_orig =None,
                            curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = None):

    # 1aplacian loss
    laplacian_loss = get_laplacian_regularization_loss(weights['W_laplacian'], mesh_orig_vertices, mesh_adv_vertices,
                                                       non_weighted_Laplacian_matrix)
    # print(laplacian_loss)

    # compute some value base on need
    # compute area and normal per vertex
    # area_per_vertex = get_area_surround_per_vertex(mesh_adv_vertices, faces)
    normal_per_face_adv = compute_face_normals(mesh_adv_vertices, faces)

    # t1 = time.time()
    # normal_per_vertex = get_normal_per_vertex(mesh_adv_vertices, faces)
    # t2 = time.time()
    # print("time for compute area-weighted vertex normals: ", t2 - t1)
    #
    # t3 = time.time()
    # normal_per_vertex_othe = get_normal_per_vertex_v2(mesh_adv_vertices, faces)
    # t4 = time.time()
    # print("time for compute average vertex normal: ", t4 - t3)
    # area_per_vertex, normal_per_vertex = get_area_surround_per_vertex_and_normal_per_vertex(mesh_adv_vertices, faces)

    # area_per_face_adv = compute_face_areas(mesh_adv_vertices, faces)
    # area_loss = get_area_loss(weights['W_area'], area_per_face_orig, area_per_face_adv,faces)
    # surround_area_loss = get_area_per_vertex_loss(weights, area_per_vertex_orig, area_per_vertex)

    # curvature loss
    # curvature_per_vertex_adv = get_curvature_per_vertex(mesh_adv_vertices, edges, normal_per_vertex)
    # curvature_loss = get_curvature_loss(weights, curvature_per_vertex_orig, curvature_per_vertex_adv, mesh_adv_vertices)

    # norm loss
    # vertex_norm_loss = get_vertex_normal_loss(weights, normal_per_vertex_orig, normal_per_vertex)
    face_norm_loss = get_face_normal_loss(weights['W_normal'], normal_per_face_orig, normal_per_face_adv)


    # l2 loss
    # l2_loss = get_l2_loss(weights, mesh_orig_vertices,  mesh_adv_vertices)
    # area_weighted_l2_loss = get_area_weighted_l2_loss(weights, mesh_orig_vertices, mesh_adv_vertices, area_per_vertex)

    # edge loss
    edge_loss = get_edge_loss(weights["W_edge"],mesh_orig_vertices, mesh_adv_vertices, faces, edge_pairs)


    # regularization_loss = laplacian_loss + area_loss + vertex_norm_loss + edge_loss + curvature_loss
    # regularization_loss = laplacian_loss + area_weighted_l2_loss + face_norm_loss + edge_loss + curvature_loss

    # print some info if needed
    print(f"Laplacian loss: {laplacian_loss},Edge loss: {edge_loss}, Norm loss: {face_norm_loss}")
    # print(f" Norm loss: {face_norm_loss}")
    # print(f"Laplacian loss: {laplacian_loss}, Curvature loss: {curvature_loss}, "
    #       f"Edge loss: {edge_loss}, Norm loss: {face_norm_loss}, Area loss:{surround_area_loss}")
    # print("lap_loss", laplacian_loss)
    # print('edge_loss', edge_loss)
    # print('norm_loss', face_norm_loss)
    # print('area_loss', area_loss)
    # print("l2_loss", l2_loss)
    # print("cirvature_loss", curvature_loss)
    # perceptual_loss = face_norm_loss
    perceptual_loss = laplacian_loss + face_norm_loss + edge_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    return perceptual_loss, face_norm_loss

def get_perceptual_loss_edge_norm(weights, mesh_orig_vertices, mesh_adv_vertices, faces, edges=None, edge_pairs =None
                            , area_per_face_orig = None, normal_per_face_orig = None,
                            area_per_vertex_orig = None, normal_per_vertex_orig =None,
                            curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = None):

    # 1aplacian loss
    laplacian_loss = get_laplacian_regularization_loss(weights['W_laplacian'], mesh_orig_vertices, mesh_adv_vertices,
                                                       non_weighted_Laplacian_matrix)
    # print(laplacian_loss)

    # compute some value base on need
    # compute area and normal per vertex
    # area_per_vertex = get_area_surround_per_vertex(mesh_adv_vertices, faces)
    normal_per_face_adv = compute_face_normals(mesh_adv_vertices, faces)

    # t1 = time.time()
    # normal_per_vertex = get_normal_per_vertex(mesh_adv_vertices, faces)
    # t2 = time.time()
    # print("time for compute area-weighted vertex normals: ", t2 - t1)
    #
    # t3 = time.time()
    # normal_per_vertex_othe = get_normal_per_vertex_v2(mesh_adv_vertices, faces)
    # t4 = time.time()
    # print("time for compute average vertex normal: ", t4 - t3)
    # area_per_vertex, normal_per_vertex = get_area_surround_per_vertex_and_normal_per_vertex(mesh_adv_vertices, faces)

    # area_per_face_adv = compute_face_areas(mesh_adv_vertices, faces)
    # area_loss = get_area_loss(weights['W_area'], area_per_face_orig, area_per_face_adv,faces)
    # surround_area_loss = get_area_per_vertex_loss(weights, area_per_vertex_orig, area_per_vertex)

    # curvature loss
    # curvature_per_vertex_adv = get_curvature_per_vertex(mesh_adv_vertices, edges, normal_per_vertex)
    # curvature_loss = get_curvature_loss(weights, curvature_per_vertex_orig, curvature_per_vertex_adv, mesh_adv_vertices)

    # norm loss
    # vertex_norm_loss = get_vertex_normal_loss(weights, normal_per_vertex_orig, normal_per_vertex)
    face_norm_loss = get_face_normal_loss(weights['W_normal'], normal_per_face_orig, normal_per_face_adv)


    # l2 loss
    # l2_loss = get_l2_loss(weights, mesh_orig_vertices,  mesh_adv_vertices)
    # area_weighted_l2_loss = get_area_weighted_l2_loss(weights, mesh_orig_vertices, mesh_adv_vertices, area_per_vertex)

    # edge loss
    edge_loss = get_edge_loss(weights["W_edge"],mesh_orig_vertices, mesh_adv_vertices, faces, edge_pairs)


    # regularization_loss = laplacian_loss + area_loss + vertex_norm_loss + edge_loss + curvature_loss
    # regularization_loss = laplacian_loss + area_weighted_l2_loss + face_norm_loss + edge_loss + curvature_loss

    # print some info if needed
    print(f"Laplacian loss: {laplacian_loss},Edge loss: {edge_loss}, Norm loss: {face_norm_loss}")
    # print(f" Norm loss: {face_norm_loss}")
    # print(f"Laplacian loss: {laplacian_loss}, Curvature loss: {curvature_loss}, "
    #       f"Edge loss: {edge_loss}, Norm loss: {face_norm_loss}, Area loss:{surround_area_loss}")
    # print("lap_loss", laplacian_loss)
    # print('edge_loss', edge_loss)
    # print('norm_loss', face_norm_loss)
    # print('area_loss', area_loss)
    # print("l2_loss", l2_loss)
    # print("cirvature_loss", curvature_loss)
    # perceptual_loss = face_norm_loss
    perceptual_loss =  face_norm_loss + edge_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    return perceptual_loss, face_norm_loss


def get_perceptual_loss_edge_lap(weights, mesh_orig_vertices, mesh_adv_vertices, faces, edges=None, edge_pairs =None
                            , area_per_face_orig = None, normal_per_face_orig = None,
                            area_per_vertex_orig = None, normal_per_vertex_orig =None,
                            curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = None):

    # 1aplacian loss
    laplacian_loss = get_laplacian_regularization_loss(weights['W_laplacian'], mesh_orig_vertices, mesh_adv_vertices,
                                                       non_weighted_Laplacian_matrix)
    # print(laplacian_loss)

    # compute some value base on need
    # compute area and normal per vertex
    # area_per_vertex = get_area_surround_per_vertex(mesh_adv_vertices, faces)
    normal_per_face_adv = compute_face_normals(mesh_adv_vertices, faces)

    # t1 = time.time()
    # normal_per_vertex = get_normal_per_vertex(mesh_adv_vertices, faces)
    # t2 = time.time()
    # print("time for compute area-weighted vertex normals: ", t2 - t1)
    #
    # t3 = time.time()
    # normal_per_vertex_othe = get_normal_per_vertex_v2(mesh_adv_vertices, faces)
    # t4 = time.time()
    # print("time for compute average vertex normal: ", t4 - t3)
    # area_per_vertex, normal_per_vertex = get_area_surround_per_vertex_and_normal_per_vertex(mesh_adv_vertices, faces)

    # area_per_face_adv = compute_face_areas(mesh_adv_vertices, faces)
    # area_loss = get_area_loss(weights['W_area'], area_per_face_orig, area_per_face_adv,faces)
    # surround_area_loss = get_area_per_vertex_loss(weights, area_per_vertex_orig, area_per_vertex)

    # curvature loss
    # curvature_per_vertex_adv = get_curvature_per_vertex(mesh_adv_vertices, edges, normal_per_vertex)
    # curvature_loss = get_curvature_loss(weights, curvature_per_vertex_orig, curvature_per_vertex_adv, mesh_adv_vertices)

    # norm loss
    # vertex_norm_loss = get_vertex_normal_loss(weights, normal_per_vertex_orig, normal_per_vertex)
    face_norm_loss = get_face_normal_loss(weights['W_normal'], normal_per_face_orig, normal_per_face_adv)


    # l2 loss
    # l2_loss = get_l2_loss(weights, mesh_orig_vertices,  mesh_adv_vertices)
    # area_weighted_l2_loss = get_area_weighted_l2_loss(weights, mesh_orig_vertices, mesh_adv_vertices, area_per_vertex)

    # edge loss
    edge_loss = get_edge_loss(weights["W_edge"],mesh_orig_vertices, mesh_adv_vertices, faces, edge_pairs)


    # regularization_loss = laplacian_loss + area_loss + vertex_norm_loss + edge_loss + curvature_loss
    # regularization_loss = laplacian_loss + area_weighted_l2_loss + face_norm_loss + edge_loss + curvature_loss

    # print some info if needed
    print(f"Laplacian loss: {laplacian_loss},Edge loss: {edge_loss}, Norm loss: {face_norm_loss}")
    # print(f" Norm loss: {face_norm_loss}")
    # print(f"Laplacian loss: {laplacian_loss}, Curvature loss: {curvature_loss}, "
    #       f"Edge loss: {edge_loss}, Norm loss: {face_norm_loss}, Area loss:{surround_area_loss}")
    # print("lap_loss", laplacian_loss)
    # print('edge_loss', edge_loss)
    # print('norm_loss', face_norm_loss)
    # print('area_loss', area_loss)
    # print("l2_loss", l2_loss)
    # print("cirvature_loss", curvature_loss)
    # perceptual_loss = face_norm_loss
    perceptual_loss = laplacian_loss + edge_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    return perceptual_loss, face_norm_loss

def get_perceptual_loss_norm_lap(weights, mesh_orig_vertices, mesh_adv_vertices, faces, edges=None, edge_pairs =None
                            , area_per_face_orig = None, normal_per_face_orig = None,
                            area_per_vertex_orig = None, normal_per_vertex_orig =None,
                            curvature_per_vertex_orig = None, non_weighted_Laplacian_matrix = None):

    # 1aplacian loss
    laplacian_loss = get_laplacian_regularization_loss(weights['W_laplacian'], mesh_orig_vertices, mesh_adv_vertices,
                                                       non_weighted_Laplacian_matrix)
    # print(laplacian_loss)

    # compute some value base on need
    # compute area and normal per vertex
    # area_per_vertex = get_area_surround_per_vertex(mesh_adv_vertices, faces)
    normal_per_face_adv = compute_face_normals(mesh_adv_vertices, faces)

    # t1 = time.time()
    # normal_per_vertex = get_normal_per_vertex(mesh_adv_vertices, faces)
    # t2 = time.time()
    # print("time for compute area-weighted vertex normals: ", t2 - t1)
    #
    # t3 = time.time()
    # normal_per_vertex_othe = get_normal_per_vertex_v2(mesh_adv_vertices, faces)
    # t4 = time.time()
    # print("time for compute average vertex normal: ", t4 - t3)
    # area_per_vertex, normal_per_vertex = get_area_surround_per_vertex_and_normal_per_vertex(mesh_adv_vertices, faces)

    # area_per_face_adv = compute_face_areas(mesh_adv_vertices, faces)
    # area_loss = get_area_loss(weights['W_area'], area_per_face_orig, area_per_face_adv,faces)
    # surround_area_loss = get_area_per_vertex_loss(weights, area_per_vertex_orig, area_per_vertex)

    # curvature loss
    # curvature_per_vertex_adv = get_curvature_per_vertex(mesh_adv_vertices, edges, normal_per_vertex)
    # curvature_loss = get_curvature_loss(weights, curvature_per_vertex_orig, curvature_per_vertex_adv, mesh_adv_vertices)

    # norm loss
    # vertex_norm_loss = get_vertex_normal_loss(weights, normal_per_vertex_orig, normal_per_vertex)
    face_norm_loss = get_face_normal_loss(weights['W_normal'], normal_per_face_orig, normal_per_face_adv)


    # l2 loss
    # l2_loss = get_l2_loss(weights, mesh_orig_vertices,  mesh_adv_vertices)
    # area_weighted_l2_loss = get_area_weighted_l2_loss(weights, mesh_orig_vertices, mesh_adv_vertices, area_per_vertex)

    # edge loss
    edge_loss = get_edge_loss(weights["W_edge"],mesh_orig_vertices, mesh_adv_vertices, faces, edge_pairs)


    # regularization_loss = laplacian_loss + area_loss + vertex_norm_loss + edge_loss + curvature_loss
    # regularization_loss = laplacian_loss + area_weighted_l2_loss + face_norm_loss + edge_loss + curvature_loss

    # print some info if needed
    print(f"Laplacian loss: {laplacian_loss},Edge loss: {edge_loss}, Norm loss: {face_norm_loss}")
    # print(f" Norm loss: {face_norm_loss}")
    # print(f"Laplacian loss: {laplacian_loss}, Curvature loss: {curvature_loss}, "
    #       f"Edge loss: {edge_loss}, Norm loss: {face_norm_loss}, Area loss:{surround_area_loss}")
    # print("lap_loss", laplacian_loss)
    # print('edge_loss', edge_loss)
    # print('norm_loss', face_norm_loss)
    # print('area_loss', area_loss)
    # print("l2_loss", l2_loss)
    # print("cirvature_loss", curvature_loss)
    # perceptual_loss = face_norm_loss
    perceptual_loss = laplacian_loss + face_norm_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    # perceptual_loss = laplacian_loss + face_norm_loss + edge_loss + curvature_loss + surround_area_loss
    return perceptual_loss, face_norm_loss
def get_adaptive_edge_loss(w,orig_vertices, adv_vertices, vertex_coefficient, faces = None, edge_pairs =None, ):
    # w = weights["W_edge"]
    orig_edge_vertices = tf.gather(orig_vertices, edge_pairs)
    adv_edge_vertices = tf.gather(adv_vertices, edge_pairs)
    orig_edge_length = tf.norm(orig_edge_vertices[:, 1] - orig_edge_vertices[:, 0], axis=-1)
    adv_edge_length = tf.norm(adv_edge_vertices[:, 1] - adv_edge_vertices[:, 0], axis=-1)
    n = edge_pairs.shape[0]
    # edge_length_diff = adv_edge_length - orig_edge_length
    # edge_loss = tf.nn.l2_loss(edge_length_diff) / n * 2

    edge_coefficient = tf.gather(vertex_coefficient, edge_pairs)
    edge_coefficient = tf.reduce_mean(edge_coefficient, axis=-1, keepdims=False)

    edge_length_diff = adv_edge_length - orig_edge_length
    edge_length_diff_squared = tf.square(edge_length_diff)

    # edge_coefficient_expanded = tf.expand_dims(edge_coefficient, axis = -1)
    # edge_coefficient_expanded = tf.tile(edge_coefficient_expanded, [1, 3])

    edge_coefficient_weighted_laplacian_diff_squared = edge_length_diff_squared * edge_coefficient
    #
    adaptive_edge_length_loss = tf.reduce_sum(edge_coefficient_weighted_laplacian_diff_squared) / n
    # 返回边缘损失乘以权重
    return adaptive_edge_length_loss * w

def get_edge_loss(w,orig_vertices, adv_vertices, faces = None, edge_pairs =None):
    # w = weights["W_edge"]
    orig_edge_vertices = tf.gather(orig_vertices, edge_pairs)
    adv_edge_vertices = tf.gather(adv_vertices, edge_pairs)
    orig_edge_length = tf.norm(orig_edge_vertices[:, 1] - orig_edge_vertices[:, 0], axis=-1)
    adv_edge_length = tf.norm(adv_edge_vertices[:, 1] - adv_edge_vertices[:, 0], axis=-1)
    edge_count = edge_pairs.shape[0]
    edge_length_diff = adv_edge_length - orig_edge_length
    edge_loss = tf.nn.l2_loss(edge_length_diff) / edge_count * 2

    # edge_count = 0
    # edge_loss = tf.constant(0.0, dtype=tf.float32)
    # for face in faces:
    #     for i in range(tf.shape(face)[0].numpy()):
    #         edge_count += 1
    #         vertex_index1 = face[i]
    #         vertex_index2 = face[(i + 1) % tf.shape(face)[0]]
    #         orig_vertex1 = tf.gather(orig_vertices, vertex_index1)
    #         orig_vertex2 = tf.gather(orig_vertices, vertex_index2)
    #
    #         adv_vertex1 = tf.gather(adv_vertices, vertex_index1)
    #         adv_vertex2 = tf.gather(adv_vertices, vertex_index2)
    #
    #         orig_edge_length = distance(orig_vertex1,orig_vertex2)
    #         adv_edge_length = distance(adv_vertex1,adv_vertex2)
    #         edge_loss = tf.add(edge_loss,tf.square(adv_edge_length - orig_edge_length))
    # edge_loss = tf.divide(edge_loss,edge_count)
    # 返回边缘损失乘以权重
    return edge_loss * w

def get_area_weighted_l2_loss(weights, mesh_vertices, adv_mesh_vertices, area_per_vertex):
    w = weights['W_reg_area']
    n = mesh_vertices.shape[0]
    mesh_diff = adv_mesh_vertices - mesh_vertices
    squared_diff = tf.square(mesh_diff)
    squared_distances = tf.reduce_sum(squared_diff, axis=1)
    area_weighted_mesh_vertex_l2_distance = squared_distances / area_per_vertex
    area_weighted_l2_loss = tf.reduce_sum(area_weighted_mesh_vertex_l2_distance) / n

    return area_weighted_l2_loss * w
def get_l2_loss(weights, mesh_vertices, adv_mesh_vertices):
    w = weights["W_l2"]
    n = mesh_vertices.shape[0]
    l2_loss = tf.nn.l2_loss(adv_mesh_vertices- mesh_vertices) * 2 / n
    return l2_loss * w

def get_vertex_normal_loss(weights, normal_per_vertex_orig, normal_per_vertex_adv):
    w = weights['W_reg_normals']
    n = normal_per_vertex_orig.shape[0]
    norm_loss = tf.nn.l2_loss(normal_per_vertex_adv - normal_per_vertex_orig) * 2 / n

    return norm_loss * w


def get_face_normal_loss(w, normal_per_face_orig, normal_per_face_adv):
    # w = weights['W_normal']
    n = normal_per_face_orig.shape[0]
    norm_loss = tf.nn.l2_loss(normal_per_face_adv - normal_per_face_orig) * 2 / n

    return norm_loss * w
def get_non_weighted_Laplacian(vertices, faces, edges, edge_pairs):
    mesh_adjacency_matrix = get_adjacency_matrix_of_mesh(vertices, faces, edge_pairs)
    # 计算网格的度矩阵,度矩阵是对角线矩阵，可以用一个[]一维tensor表示
    # 为了简化计算
    mesh_inverse_degree_matrix = get_inverse_degree_matrix_of_mesh(vertices, faces, edges)
    mesh_inverse_degree_matrix_dense = tf.sparse.to_dense(mesh_inverse_degree_matrix)
    Dinverse_J_matrix = tf.sparse.sparse_dense_matmul(mesh_inverse_degree_matrix_dense,mesh_adjacency_matrix)
    unit_matrix = tf.eye(vertices.shape[0], dtype=tf.float32)
    non_matrix = unit_matrix - Dinverse_J_matrix
    return non_matrix


def get_laplacian_regularization_loss(w, mesh_vertices, adv_mesh_vertices, non_weighted_Laplacian_matrix):

    # mesh_degree_matrix = get_degree_matrix_of_mesh(mesh_vertices, faces)
    # adv_mesh_degree_matrix = get_degree_matrix_of_mesh(mesh_vertices, faces)
    # w = weights['W_laplacian']
    n = mesh_vertices.shape[0]
    # orig_non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(mesh_vertices, faces, edges, edge_pairs)
    # adv_non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(adv_mesh_vertices, faces, edges, edge_pairs)
    orig_laplacian_vertices = tf.matmul(non_weighted_Laplacian_matrix, mesh_vertices)
    adv_laplacian_vertices = tf.matmul(non_weighted_Laplacian_matrix, adv_mesh_vertices)

    laplacian_diff = adv_laplacian_vertices - orig_laplacian_vertices
    laplacian_regularization_loss = tf.nn.l2_loss(laplacian_diff) / n * 2

    return laplacian_regularization_loss * w

def get_adaptive_laplacian_regularization_loss(w, mesh_vertices, adv_mesh_vertices,
                                               non_weighted_Laplacian_matrix, vertex_coefficient):

    # mesh_degree_matrix = get_degree_matrix_of_mesh(mesh_vertices, faces)
    # adv_mesh_degree_matrix = get_degree_matrix_of_mesh(mesh_vertices, faces)
    # w = weights['W_laplacian']
    n = mesh_vertices.shape[0]
    # orig_non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(mesh_vertices, faces, edges, edge_pairs)
    # adv_non_weighted_Laplacian_matrix = get_non_weighted_Laplacian(adv_mesh_vertices, faces, edges, edge_pairs)
    orig_laplacian_vertices = tf.matmul(non_weighted_Laplacian_matrix, mesh_vertices)
    adv_laplacian_vertices = tf.matmul(non_weighted_Laplacian_matrix, adv_mesh_vertices)

    laplacian_diff = adv_laplacian_vertices - orig_laplacian_vertices
    laplacian_diff_squared = tf.square(laplacian_diff)
    vertex_coefficient_expanded = tf.expand_dims(vertex_coefficient, axis = -1)
    vertex_coefficient_expanded = tf.tile(vertex_coefficient_expanded, [1, 3])

    vertex_coefficient_weighted_laplacian_diff_squared = laplacian_diff_squared * vertex_coefficient_expanded

    adaptive_laplacian_regularization_loss = tf.reduce_sum(vertex_coefficient_weighted_laplacian_diff_squared) / n
    # laplacian_regularization_loss = tf.nn.l2_loss(laplacian_diff) / n * 2
    return adaptive_laplacian_regularization_loss * w



def get_inverse_degree_matrix_of_mesh(vertices, faces, edges):
    vertices = tf.cast(vertices,dtype=tf.float32)
    vertices_number = vertices.shape[0]

    degrees = np.sum(edges != -1,axis = 1,dtype=np.float32)

    reciprocal_degrees = np.reciprocal(degrees)
    reciprocal_degrees = tf.convert_to_tensor(reciprocal_degrees, dtype=tf.float32)
    # 构建稀疏矩阵的行、列、和数据
    rows = np.arange(len(degrees))
    cols = np.arange(len(degrees))
    indices = np.column_stack((rows, cols))
    indices = tf.constant(indices)
    data = reciprocal_degrees
    # 使用稀疏矩阵的 coo_matrix 构造器
    degree_matrix_sparse = tf.sparse.SparseTensor(indices = indices, values=data,
                                                  dense_shape=(vertices_number, vertices_number))
    # 将 coo_matrix 转换为 csr_matrix，以便于后续计算
    # degree_matrix_sparse = tf.sparse.reorder(degree_matrix_sparse)

    # shape = tf.constant((vertices_number,vertices_number),dtype=tf.int32)
    # adjacency_matrix = tf.zeros((num_vertices, num_vertices), dtype=tf.float32)

    # for face in faces:
    #     indices = tf.constant(face, dtype=tf.int32)
    #     updates = tf.ones((tf.shape(indices)[0],), dtype=tf.float32)  # 更新为一个与面片顶点数相匹配的向量
    #     adjacency_matrix = tf.tensor_scatter_nd_add(adjacency_matrix, tf.expand_dims(indices, axis=1), updates)
    #
    # # 构建度矩阵
    # degree_vector = tf.reduce_sum(adjacency_matrix, axis=1)
    # degree_matrix = tf.linalg.diag(degree_vector)

    return degree_matrix_sparse
def get_adjacency_matrix_of_mesh(vertices, faces,edge_pairs):

    # indices = np.array([[i, j] for face in faces for i in face for j in face if i != j])
    # 构建邻接矩阵
    indices = edge_pairs

    vertices_number = vertices.shape[0]
    values = tf.ones(indices[:, 0].shape, dtype=tf.float32)
    shape = tf.constant([vertices_number,vertices_number], dtype=tf.int64)
    adjacency_matrix_sparse = tf.sparse.SparseTensor(indices, values, shape)
    # adjacency_matrix_sparse = tf.where(tf.math.greater(adjacency_matrix_sparse, 0), 1, 0)

    return adjacency_matrix_sparse
    # 如果希望邻接矩阵是 0 或 1 的整数矩阵，可以进行以下处理

def get_curvature_loss(weights, vertex_curvature_orig, vertex_curvature_adv, mesh_adv_vertices):
    w = weights['W_reg_curvature']
    n = mesh_adv_vertices.shape[0]
    vertex_curvature_diff = vertex_curvature_adv - vertex_curvature_orig
    curvature_loss = tf.nn.l2_loss(vertex_curvature_diff) / n * 2
    return w * curvature_loss

def get_curvature_loss_v2(weights, vertex_curvature_orig, vertex_curvature_adv, mesh_adv_vertices):
    w = weights['W_reg_curvature']
    n = mesh_adv_vertices.shape[0]
    vertex_curvature_diff = vertex_curvature_adv - vertex_curvature_orig
    curvature_loss = tf.nn.l2_loss(vertex_curvature_diff) / n * 2
    return w * curvature_loss

def get_curvature_per_vertex(vertices, edges, vertex_normals):
    num_vertices = tf.shape(vertices)[0]
    # Initial
    # vertex_curvatures = tf.zeros([num_vertices], dtype=tf.float32)
    # vertex_curvatures_to_add = tf.zeros([num_vertices], dtype=tf.float32)
    n = num_vertices.numpy()
    # vertex_curvatures_split = tf.split(vertex_curvatures, int(n), axis=0)
    # vertex_curvatures_to_add_split = tf.split(vertex_curvatures_to_add, int(n), axis=0)
    to_add_list = []
    for i in range(num_vertices):
        curr_vertex_normal = tf.gather(vertex_normals, i)
        # edge_count = edges[i]
        i_edges_indices = tf.where(tf.reduce_any(tf.equal(edges, i), axis=1))
        i_edges_indices = tf.squeeze(i_edges_indices)
        i_edge_end_vertices = tf.gather(vertices, i_edges_indices)
        curr_vertex = tf.gather(vertices, i)

        # vertor: start point (i) end point neibor
        i_edge_vertors = i_edge_end_vertices - curr_vertex

        curr_vertex_normal_expended = tf.expand_dims(curr_vertex_normal, axis=1)
        inner = tf.matmul(i_edge_vertors, curr_vertex_normal_expended)
        new_vertex_curvature = tf.squeeze(tf.reduce_mean(inner, axis=0))
        to_add_list.append(new_vertex_curvature)
        # vertex_curvatures_split[i] = new_vertex_curvature
        # vertex_curvatures = tf.tensor_scatter_nd_update(vertex_curvatures, i, new_value)
        # vertex_curvatures_array.write(i, tf.squeeze(tf.reduce_mean(inner, axis=0)))
    to_add = tf.stack(to_add_list, axis=0)
    # concatenated_result = tf.concat(vertex_curvatures_to_add_split, axis=0)
    # vertex_curvatures = vertex_curvatures_array.stack()
    # vertex_curvatures_another = vertex_curvatures_to_add + to_add
    return to_add

def get_curvature_per_vertex_2(vertices, faces):

    curr_mesh = trimesh.Trimesh(vertices, faces)
    curr_mesh.face_adjacency_angles

    return

def get_curvature_per_vertex_v2(vertices, edges, vertex_normals):
    num_vertices = tf.shape(vertices)[0] # tensor shape: ()
    n = num_vertices.numpy() # int
    # Initial
    vertex_curvatures = tf.zeros([num_vertices], dtype=tf.float32)
    vertex_curvatures = tf.Variable(vertex_curvatures)


    to_add_list = []
    for i in range(num_vertices):
        curr_vertex_normal = tf.gather(vertex_normals, i)
        # edge_count = edges[i]
        i_edges_indices = tf.where(tf.reduce_any(tf.equal(edges, i), axis=1))
        i_edges_indices = tf.squeeze(i_edges_indices)
        i_edge_end_vertices = tf.gather(vertices, i_edges_indices)
        curr_vertex = tf.gather(vertices, i)
        # vertor: start point (i) end point neibor
        i_edge_vertors = i_edge_end_vertices - curr_vertex
        curr_vertex_normal_expended = tf.expand_dims(curr_vertex_normal, axis=1)
        inner = tf.matmul(i_edge_vertors, curr_vertex_normal_expended)
        new_vertex_curvature = tf.squeeze(tf.reduce_mean(inner, axis=0))
        vertex_curvatures = tf.tensor_scatter_nd_update(vertex_curvatures, indices=[[i]], updates=[new_vertex_curvature])
        # to_add_list.append(new_vertex_curvature)
        # vertex_curvatures_split[i] = new_vertex_curvature
        # vertex_curvatures = tf.tensor_scatter_nd_update(vertex_curvatures, i, new_value)
        # vertex_curvatures_array.write(i, tf.squeeze(tf.reduce_mean(inner, axis=0)))
    # to_add = tf.stack(to_add_list, axis=0)
    # concatenated_result = tf.concat(vertex_curvatures_to_add_split, axis=0)
    # vertex_curvatures = vertex_curvatures_array.stack()
    # vertex_curvatures_another = vertex_curvatures_to_add + to_add
    return vertex_curvatures


def get_area_surround_per_vertex_and_normal_per_vertex(vertices, faces):
    num_vertices = tf.shape(vertices)[0]
    # area_surround_per_vertex_array = tf.TensorArray(dtype=tf.float32, size=num_vertices)
    # normal_per_vertex_array = tf.TensorArray(dtype=tf.float32, size=num_vertices)
    area_surround_per_vertex_list = []
    normal_per_vertex_list = []
    for i in range(num_vertices):
        # 找到包含顶点 i 的所有三角面的索引
        i_faces_indices = tf.where(tf.reduce_any(tf.equal(faces, i), axis=1))
        i_faces_indices = tf.squeeze(i_faces_indices)

        # if tf.shape(i_faces_indices)[0] == 0:
        #     # 如果没有包含顶点 i 的面，则设置面积为 0，法向量为 0 向量
        #     area_surround_per_vertex_array.write(i, 0.0)
        #     normal_per_vertex_array.write(i, tf.zeros([3]))
        #     continue

        i_faces = tf.gather(faces, i_faces_indices)

        # 提取包含顶点 i 的面片的顶点坐标
        i_faces_vertices = tf.gather(vertices, i_faces)

        # 计算每个面的法向量和面积
        edge1 = i_faces_vertices[:, 1, :] - i_faces_vertices[:, 0, :]
        edge2 = i_faces_vertices[:, 2, :] - i_faces_vertices[:, 0, :]
        face_normals = tf.linalg.cross(edge1, edge2)
        face_areas = 0.5 * tf.norm(face_normals, axis=1)

        # 计算顶点 i 的总面积和面积加权法向量
        total_area = tf.reduce_sum(face_areas)
        face_areas = tf.expand_dims(face_areas, axis=1)
        area_weighted_normals = face_areas * face_normals
        normal_per_vertice = tf.reduce_sum(area_weighted_normals, axis=0)
        normal_per_vertice_normalized = tf.math.l2_normalize(normal_per_vertice, axis=0)

        # 存储顶点 i 的总面积和法向量
        area_surround_per_vertex_list.append(total_area)
        normal_per_vertex_list.append(normal_per_vertice_normalized)
        # area_surround_per_vertex_array.write(i, total_area)
        # normal_per_vertex_array.write(i, normal_per_vertice_normalized)
    # area_surround_per_vertex_another = area_surround_per_vertex_array.stack()
    # normal_per_vertex_another = normal_per_vertex_array.stack()
    area_surround_per_vertex = tf.stack(area_surround_per_vertex_list, axis=0)
    normal_per_vertex = tf.stack(normal_per_vertex_list, axis=0)

    return area_surround_per_vertex, normal_per_vertex



def get_area_surround_per_vertex(vertices, faces):
    num_vertices = tf.shape(vertices)[0]
    area_surround_per_vertex_list = []
    for i in range(num_vertices):
        # 找到包含顶点 i 的所有三角面的索引
        i_faces_indices = tf.where(tf.reduce_any(tf.equal(faces, i), axis=1))
        i_faces_indices = tf.squeeze(i_faces_indices)

        i_faces = tf.gather(faces, i_faces_indices)
        # 提取包含顶点 i 的面片的顶点坐标
        i_faces_vertices = tf.gather(vertices, i_faces)

        # 计算每个面的法向量和面积
        edge1 = i_faces_vertices[:, 1, :] - i_faces_vertices[:, 0, :]
        edge2 = i_faces_vertices[:, 2, :] - i_faces_vertices[:, 0, :]
        face_normals = tf.linalg.cross(edge1, edge2)
        face_areas = 0.5 * tf.norm(face_normals, axis=1)

        # 计算顶点 i 的总面积和面积加权法向量
        total_area = tf.reduce_sum(face_areas)
        # face_areas = tf.expand_dims(face_areas, axis=1)
        # area_weighted_normals = face_areas * face_normals
        # normal_per_vertice = tf.reduce_sum(area_weighted_normals, axis=0)
        # normal_per_vertice_normalized = tf.math.l2_normalize(normal_per_vertice, axis=0)

        # 存储顶点 i 的总面积和法向量
        area_surround_per_vertex_list.append(total_area)
        # normal_per_vertex_list.append(normal_per_vertice_normalized)
        # area_surround_per_vertex_array.write(i, total_area)
        # normal_per_vertex_array.write(i, normal_per_vertice_normalized)
    # area_surround_per_vertex_another = area_surround_per_vertex_array.stack()
    # normal_per_vertex_another = normal_per_vertex_array.stack()
    area_surround_per_vertex = tf.stack(area_surround_per_vertex_list, axis=0)
    # normal_per_vertex = tf.stack(normal_per_vertex_list, axis=0)

    return area_surround_per_vertex


def get_normal_per_vertex(vertices, faces):
    num_vertices = tf.shape(vertices)[0]
    # area_surround_per_vertex_array = tf.TensorArray(dtype=tf.float32, size=num_vertices)
    # normal_per_vertex_array = tf.TensorArray(dtype=tf.float32, size=num_vertices)
    # area_surround_per_vertex_list = []
    normal_per_vertex_list = []

    for i in range(num_vertices):
        # 找到包含顶点 i 的所有三角面的索引
        i_faces_indices = tf.where(tf.reduce_any(tf.equal(faces, i), axis=1))
        i_faces_indices = tf.squeeze(i_faces_indices)

        # if tf.shape(i_faces_indices)[0] == 0:
        #     # 如果没有包含顶点 i 的面，则设置面积为 0，法向量为 0 向量
        #     area_surround_per_vertex_array.write(i, 0.0)
        #     normal_per_vertex_array.write(i, tf.zeros([3]))
        #     continue

        i_faces = tf.gather(faces, i_faces_indices)

        # 提取包含顶点 i 的面片的顶点坐标
        i_faces_vertices = tf.gather(vertices, i_faces)

        # 计算每个面的法向量和面积
        edge1 = i_faces_vertices[:, 1, :] - i_faces_vertices[:, 0, :]
        edge2 = i_faces_vertices[:, 2, :] - i_faces_vertices[:, 0, :]
        face_normals = tf.linalg.cross(edge1, edge2)
        face_areas = 0.5 * tf.norm(face_normals, axis=1)

        # 计算顶点 i 的总面积和面积加权法向量
        total_area = tf.reduce_sum(face_areas)
        face_areas = tf.expand_dims(face_areas, axis=1)
        area_weighted_normals = face_areas * face_normals
        normal_per_vertice = tf.reduce_sum(area_weighted_normals, axis=0)
        normal_per_vertice_normalized = tf.math.l2_normalize(normal_per_vertice, axis=0)

        # 存储顶点 i 的总面积和法向量
        # area_surround_per_vertex_list.append(total_area)
        normal_per_vertex_list.append(normal_per_vertice_normalized)
        # area_surround_per_vertex_array.write(i, total_area)
        # normal_per_vertex_array.write(i, normal_per_vertice_normalized)
    # area_surround_per_vertex_another = area_surround_per_vertex_array.stack()
    # normal_per_vertex_another = normal_per_vertex_array.stack()
    # area_surround_per_vertex = tf.stack(area_surround_per_vertex_list, axis=0)
    normal_per_vertex = tf.stack(normal_per_vertex_list, axis=0)

    return normal_per_vertex

def get_normal_per_vertex_v2(vertices, faces):
    # no area_weighted
    num_vertices = tf.shape(vertices)[0]
    normal_per_vertex_list = []
    for i in range(num_vertices):
        # 找到包含顶点 i 的所有三角面的索引
        i_faces_indices = tf.where(tf.reduce_any(tf.equal(faces, i), axis=1))
        i_faces_indices = tf.squeeze(i_faces_indices)

        # if tf.shape(i_faces_indices)[0] == 0:
        #     # 如果没有包含顶点 i 的面，则设置面积为 0，法向量为 0 向量
        #     area_surround_per_vertex_array.write(i, 0.0)
        #     normal_per_vertex_array.write(i, tf.zeros([3]))
        #     continue

        i_faces = tf.gather(faces, i_faces_indices)

        # 提取包含顶点 i 的面片的顶点坐标
        i_faces_vertices = tf.gather(vertices, i_faces)

        # 计算每个面的法向量和面积
        edge1 = i_faces_vertices[:, 1, :] - i_faces_vertices[:, 0, :]
        edge2 = i_faces_vertices[:, 2, :] - i_faces_vertices[:, 0, :]
        face_normals = tf.linalg.cross(edge1, edge2)
        # face_areas = 0.5 * tf.norm(face_normals, axis=1)

        # 计算顶点 i 的总面积和面积加权法向量
        # total_area = tf.reduce_sum(face_areas)
        # face_areas = tf.expand_dims(face_areas, axis=1)
        # area_weighted_normals = face_areas * face_normals
        normal_per_vertice = tf.reduce_sum(face_normals, axis=0)
        normal_per_vertice_normalized = tf.math.l2_normalize(normal_per_vertice, axis=0)

        # 存储顶点 i 的总面积和法向量
        # area_surround_per_vertex_list.append(total_area)
        normal_per_vertex_list.append(normal_per_vertice_normalized)
        # area_surround_per_vertex_array.write(i, total_area)
        # normal_per_vertex_array.write(i, normal_per_vertice_normalized)
    # area_surround_per_vertex_another = area_surround_per_vertex_array.stack()
    # normal_per_vertex_another = normal_per_vertex_array.stack()
    # area_surround_per_vertex = tf.stack(area_surround_per_vertex_list, axis=0)
    normal_per_vertex = tf.stack(normal_per_vertex_list, axis=0)

    return normal_per_vertex


def distance(vertex1, vertex2):

    return tf.sqrt(tf.reduce_sum(tf.square(vertex1 - vertex2)))

def threshold_loss(W, W0, weight):

    return weight * tf.nn.l2_loss(W, W0)




if __name__ == '__main__':
    dataset_clean_path = '/home/kang/SSD/datasets/shrec_16_f500'
    dataset_adv_path = '//attack_results/RW_PDMeshNet_attack_bound0.04/aa_last_npzs/RW_PD_bound0.04'
    output_file = "evaluate_results/Ours_perceptual_loss.txt"