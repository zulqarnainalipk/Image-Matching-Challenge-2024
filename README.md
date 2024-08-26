
# 🌟 Image Matching Challenge 2024 - Hexathlon 🌟

## Overview
Welcome, brave adventurers, to the thrilling Image Matching Challenge 2024 - Hexathlon! Here, amidst the vast expanse of digital landscapes, your mission, should you choose to accept it, is to reconstruct 3D scenes from 2D images across six distinct domains. From ancient ruins to bustling city streets, from serene forests to the twinkling night sky, each domain presents its own set of challenges to overcome.

Last year's Image Matching Challenge was but a prelude to this grand spectacle. This year, stakes are raised, weaving together an intricate tapestry of diverse scenarios into a single competition. Let's  test our mettle and push the boundaries of computer vision

## 🛠️ Setup Environment
Prepare yourselves, intrepid explorers, for the journey ahead requires meticulous preparation. Fear not, for we shall guide you through the arcane rituals of environment setup with clarity and wit:

1. **Install Necessary Libraries**: 📚
   To equip your arsenal with the tools needed for this quest, execute the following commands:

   ```bash
   !pip install -r /kaggle/input/check-image-orientation/requirements.txt
   !pip install --no-index /kaggle/input/imc2024-packages-lightglue-rerun-kornia/* --no-deps
   ```

2. **Setup Cache and Checkpoints**: 🔒
   Fortify your cache and checkpoints with the resilience of ancient guardians:

   ```bash
   !mkdir -p /root/.cache/torch/hub/checkpoints
   !cp /kaggle/input/aliked/pytorch/aliked-n16/1/* /root/.cache/torch/hub/checkpoints/
   !cp /kaggle/input/lightglue/pytorch/aliked/1/* /root/.cache/torch/hub/checkpoints/
   !cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth
   !cp /kaggle/input/check-image-orientation/2020-11-16_resnext50_32x4d.zip /root/.cache/torch/hub/checkpoints/
   ```

3. **Import Essential Libraries**: 📦
   Arm yourselves with the knowledge and power of the ancients with these sacred incantations:

   ```python
   import libraries
   from pathlib import Path
   from copy import deepcopy
   import numpy as np
   import math
   import pandas as pd
   import pandas.api.types
   from itertools import combinations
   import sys, torch, h5py, pycolmap, datetime
   from PIL import Image
   from pathlib import Path
   import torch.nn.functional as F
   import torchvision.transforms.functional as TF
   import kornia as K
   import kornia.feature as KF
   from lightglue.utils import load_image
   from lightglue import LightGlue, ALIKED, match_pair
   from transformers import AutoImageProcessor, AutoModel
   from check_orientation.pre_trained_models import create_model
   sys.path.append("/kaggle/input/colmap-db-import")
   import sqlite3
   import os, argparse, h5py, warnings
   import numpy as np
   from tqdm import tqdm
   from PIL import Image, ExifTags
   from database import COLMAPDatabase, image_ids_to_pair_id
   from h5_to_db import *
   ```

Now, adventurers, with your environment fortified and your libraries in hand, you stand poised at the precipice of discovery, ready to delve into the depths of image matching and registration!

## 🧩 Concepts Explored
Behold, noble souls, the sacred knowledge that shall guide you on your quest:

1. **Feature Matching**: 🌟
   Traverse the realm of feature matching, where keypoints align and images harmonize through the magic of computer vision.

2. **RANSAC (Random Sample Consensus)**: 🎲
   Embrace the randomness of RANSAC, a robust method that triumphs over outliers and guides you on the path to accurate image registration.

3. **Sparse Reconstruction**: 🌌
   Witness the reconstruction of 3D scenes from sparse image correspondences, a feat achieved through the enigmatic algorithms of sparse reconstruction.

4. **Mean Average Accuracy (mAA)**: 🎯
   Gauge the accuracy of your endeavors with the noble metric of mAA, measuring the alignment of images with a precision fit for champions.

5. **Homogeneous Transformation Matrix**: 🔄
   Let the homogeneous transformation matrix be your guide through the labyrinth of rigid transformations, as you align images and estimate camera poses with finesse.

6. **Affine Transformation**: 🖌️
   Marvel at the versatility of affine transformations, shaping images with the strokes of a digital brush to correct distortions and align perspectives.

7. **Quaternion Representation**: 🔮
   Peer into the depths of quaternion representation, where rotations in 3D space unfold with elegance and grace, free from the shackles of gimbal lock.

## 👉 [View Notebook on Kaggle](https://www.kaggle.com/code/zulqarnainalipk/imc-24-explained/)


## 🌟 Keep Exploring! 🌟

Thanks a bunch for diving into this notebook! If you had a blast or learned something new, why not dive into more of my captivating projects and contributions on my profile?

👉 [Let's Explore More!](https://www.kaggle.com/zulqarnainalipk) 👈

[GitHub](https://github.com/zulqarnainalipk) |
[LinkedIn](https://www.linkedin.com/in/zulqarnainalipk/)

## 💬 Share Your Thoughts! 💡

Your feedback is like treasure to us! Your brilliant ideas and insights fuel our ongoing improvement. Got something to say, ask, or suggest? Don't hold back!

📬 Drop me a line via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)

Huge thanks for your time and engagement. Your support is like rocket fuel propelling me to create even more epic content.
Keep coding joyfully and wishing you stellar success in your data science adventures! 🚀

---
