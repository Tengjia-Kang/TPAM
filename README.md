# TPAM in TensorFlow
### Pacific Graphics 2024
Transferable Perceptual-constrained Adversarial Meshes (TPAM) is a method for generating adversarial meshes that disrupt the predictions of mesh classification networks, ensuring stealthiness, and possessing transferability to various mesh classifiers.![](https://tengjia-kang-research.oss-cn-beijing.aliyuncs.com/TPAM/figs/geometry_details_v2.png)

# Getting Started
### Installation
- Clone this repository:
```bash
git clone https://github.com/Tengjia-Kang/TPAM.git
cd TPAM
```
-  Create a virtual environment.

```bash
conda create -n TPAM python=3.8
conda activate TPAM
pip install -r requirements.txt
```

### Pipeline

Our attack is in a black setting, it could be divided into two part in our pipeline.

Firstly, training a surrogate network for the target victim classifier.

Secondly, attack surrogate network to obtain adversarial meshes that could transfer attack target models.

### Data

#### Raw datasets

The datasets we use are consistent with the datasets used in the mesh classification we aim to attack.

To get the raw datasets go to the relevant website.

[MeshCNN](https://github.com/ranahanocka/MeshCNN.git)
[MeshNet](https://github.com/iMoonLab/MeshNet.git)
[PD-MeshNet](https://github.com/MIT-SPARK/PD-MeshNet.git)
[MeshWalker](https://github.com/AlonLahav/MeshWalker.git)
[RIMeshGNN](https://github.com/BSResearch/RIMeshGNN.git)
[SubdivNet](https://github.com/Tengjia-Kang/SubdivNet.git)
[ExMeshCNN](https://github.com/gyeomo/ExMeshCNN.git)

#### Processed datasets

Organize the dataset with the predict logits from the target classifier.

```sh
python data/dataset_prepare.py shrec11
```

The predictions obtained by querying the target network. For different attack targets, you may need to implement different scripts, you can refer to the fork version in my repositories.

### Surrogate network train

train a surrogate network for target.

```sh
python surrogate_train/imitating_network_train.py
```

### Attack

```sh
python attack/attack_mesh.py
```

# Questions / Issues
If you have any questions or issues running this code, please open an issue so we can know to fix it, or send a email to author.

# Acknowledgments
This code design was adopted from [Random-Walks-for-Adversarial-Meshes
Public](https://github.com/amirbelder/Random-Walks-for-Adversarial-Meshes.git).

