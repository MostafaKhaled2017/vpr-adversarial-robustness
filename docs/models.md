# Models

## SuperVLAD checkpoint

**Purpose**

Base visual place recognition checkpoint for training resume and evaluation.

**Files**

- `checkpoints/SuperVLAD.pth`

**Inputs**

Normalized RGB image tensors prepared by SuperVLAD transforms.

**Outputs**

Image descriptors used for retrieval.

**Notes**

The checkpoint path appears in command examples but the file is not listed in the repository.

## Perceptual adversarial checkpoint

**Purpose**

Checkpoint produced by or used for perceptual adversarial training comparisons.

**Files**

- `checkpoints/perceptual_adv_checkpoint.pth`

**Inputs**

Normalized RGB image tensors.

**Outputs**

Image descriptors used for retrieval.

**Notes**

The checkpoint path appears in command examples but the file is not listed in the repository.

## DINOv2 ViT-B/14 foundation checkpoint

**Purpose**

Foundation model checkpoint for the SuperVLAD DINO backbone.

**Files**

- `checkpoints/dinov2_vitb14_pretrain.pth`

**Inputs**

Used during model initialization.

**Outputs**

Backbone weights for SuperVLAD model construction.

**Notes**

The SuperVLAD README links to the upstream DINOv2 checkpoint.
