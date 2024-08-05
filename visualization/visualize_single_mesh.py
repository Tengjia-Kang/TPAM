import trimesh
from src import walks_standalone

mesh_path = "/home/kang/SSD/datasets/shrec_16_f500/gorilla/test/T478.obj"
mesh = trimesh.load_mesh(mesh_path)
mesh_data = {'vertices': mesh.vertices, 'faces': mesh.faces, 'n_vertices': mesh.vertices.shape[0]}


# utils.visualize_mesh_and_pc(vertices, faces)
walks_standalone.prepare_edges_and_kdtree(mesh_data)
walk, jumps = walks_standalone.get_seq_random_walk_no_jumps(mesh_data, f0=0, seq_len=400)
walks_standalone.visualize_model(mesh_data['vertices'], mesh_data['faces'], line_width=1, show_edges=1,
                                 walk=walk, edge_colors='red')


