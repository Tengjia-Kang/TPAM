from pyvista import examples
# dataset = examples.download_saddle_surface()
# dataset.plot()
import os
import pyvista as pv
import os
# 加载一个mesh文件，这里以STL文件为例
# 你可以根据需要替换为你的mesh文件路径
# mesh

def visualize_single_mesh(mesh_path, save_folder):
    mesh_path_without_expansion = mesh_path[:-4]
    mesh_path_split_list = mesh_path_without_expansion.split('/')
    bound_name = mesh_path_split_list[-1]
    save_name = mesh_path_split_list[-3] + '_' + mesh_path_split_list[-2] + '_' +bound_name + '.png'
    save_path = os.path.join(save_folder, save_name)
    mesh = pv.read(mesh_path)
    # 创建一个绘图器实例
    # window_size = [512, 512]
    plotter = pv.Plotter()
    # plotter = pv.Plotter(off_screen=1, window_size=(int(window_size[0]), int(window_size[1])))
    plotter.set_background("#AAAAAA")
    plotter.add_mesh(mesh, color='white', opacity=1.0, edges = True , edge_color='black', line_width=100)
    plotter.show(auto_close=False)
    plotter.screenshot(save_path)
    plotter.close()


# mesh_path = "/home/kang/SSD/Projects/TPAM/attack_results/Visualization/RW/armadillo_55/L2_bound_0.001.obj"
# mesh = pv.read(mesh_path)





if __name__ == "__main__":
    save_folder = 'image_save/final'
    shrec_11_clean_folder = '/home/kang/SSD/datasets/shrec16'

    # rabait_541 = os.path.join(shrec_11_clean_folder,'rabbit','test', '541.obj')
    # visualize_single_mesh(rabait_541, save_folder)
    adv_path = "/media/kang/8E6DB583DECFBF53/AA_Ubuntu/AdvMesh/attack_results/RIMeshGNN/RIMeshGNN_Ours/aa_L2_bound_0.003/objs/spiders/test/spiders_T566_L2_boun.obj"
    clean_path = "/home/kang/SSD/datasets/shrec16/cat/test/555.obj"
    prefexx = '/home/kang/SSD/Projects/TPAM/attack_results/Visualization'
    method = 'Ours'
    mesh_name = 'cat_555'
    bound_name = "L2_bound_0.0004.obj"
    mesh_fold_path = os.path.join(prefexx, method, mesh_name)
    mesh_path = os.path.join(mesh_fold_path, bound_name)
    # visualize_single_mesh(clean_path, save_folder)
    visualize_single_mesh(adv_path, save_folder)
    # visualize_single_mesh(mesh_path, save_folder)
    # mesh_path = "/home/kang/SSD/Projects/TPAM/attack_results/Visualization/Ours/armadillo_55/L2_bound_0.001.obj"

