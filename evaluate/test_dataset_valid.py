import numpy as np
import glob
from data.dataset_prepare import shrec11_labels
import os

# shrec11 label
shrec11_labels.sort()

dataset_path = "//datasets/datasets_processed/shrec16_with_MeshWalker_pred_to_train_imitaingNet"

train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

filse = glob.glob("//datasets/datasets_processed/shrec16_with_MeshWalker_pred_to_train_imitaingNet/test/*npz")
count = 0
for file in filse:
    mesh_data = np.load(file)
    if file.split("_")[-5] == 'balls' or file.split("_")[-5] == 'ske':
        class_name = file.split("_")[-6] + '_' + file.split("_")[-5]
    else:
        class_name = file.split("_")[-5]
    index = shrec11_labels.index(class_name)
    label = mesh_data['label']
    # pred_vector = mesh_data['pred_vector']
    if(index==label):
        count += 1

print(count)