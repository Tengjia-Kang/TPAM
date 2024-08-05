from scripts import reorganize_attacked
from attack import bound
from evaluate import get_mean_curvature_Ours

if __name__ == "__main__":
    dataset_name = 'shrec11'
    path_to_adversarial_datasets = "/home/kang/M2SSD/attack_results/MeshCNN/MeshCNN_Surrogate_IterOpti_withoutBeayesian_attack_2000iter_exit20"

    dataset_adv_path = reorganize_attacked.function(path_to_adversarial_datasets)
    dataset_clean_path = '/home/kang/SSD/datasets/shrec_16_f500'
    weights =  {"W_edge": 1, 'W_normal': 0.1, 'W_laplacian': 10}

    path_union_base = dataset_adv_path.split('/')[-3]
    path_to_save_L2_distances = "evaluate_results" + '/' + "L2_norm" + '/' + path_union_base + '_l2_distances.txt'
    path_to_save_perceptuals = "evaluate_results" + '/' + "perceptual_loss" + '/' + path_union_base + '_perceptuals.txt'
    path_to_save_curvature = "evaluate_results" + '/' + "curvature" + '/' + path_union_base + '_curvature.txt'
    bound.save_L2_distance_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path,
                                                                 path_to_save_L2_distances
                                                                 , dataset_name)
    bound.save_perceptual_loss_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path,
                                                                     path_to_save_perceptuals,
                                                                     dataset_name, weights)
    get_mean_curvature_Ours.save_L2_distance_between_adversaries_and_original_mesh(dataset_clean_path, dataset_adv_path, path_to_save_curvature
                                                                                   , dataset_name)