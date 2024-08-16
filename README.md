In my view, former models had three limitations. First, convolution operation fails to use long-path information and multi-scale structural information. Second, existing models typically separate the feature extraction and fusion into two independent steps, which might leads to bad robustness. Third, transposed convolution for upsampling causes a checkerboard noise in the fused images and disrupts its quality.

I tried to tackle the first problem by fully utilizing shifted window attention mechanism along with the skip-connection of U-Net to get rid of convolution. For the second one, I alternatively using self attention and cross attention operation, which integrates feature extraction and fusion to a single unified process. For the third problem I raise an anti patch merging operation to avoid noise.

The model shows a symmetric structure and performs better than many previous models across multiple metrics. 
