# Getting Started

### Installation



# Pipeline

Our attack is in a black setting, it could be divided into two part in our pipeline.

Firstly, training a surrogate network for the target victim classifier.

Secondly, attack surrogate network to obtain adversarial meshes that could transfer attack target models.

# Surrogate network train

Training the surrogate network requires using the same version of the dataset as the target network and the corresponding ground truth. The predictions obtained by querying the target network can be used as the ground truth, and the KL divergence is selected as the training loss accordingly. If using the true categories of the mesh as the ground truth (gt), it is recommended to convert the labels into one-hot vectors and use KL divergence as the loss, or use cross-entropy loss. For query-based attacks, you need to obtain the logits for the relevant dataset from the target network, which requires your own implementation in their envs or you can use the output-logits script in our fork version.





## Attack

