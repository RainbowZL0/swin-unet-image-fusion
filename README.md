In my view, former models had three limitations. First, convolution operation fails to use long-path information and multi-scale structural information. Second, existing models typically separate the feature extraction and fusion into two independent steps, which might leads to bad robustness. Third, transposed convolution for upsampling causes a checkerboard noise in the fused images and disrupts its quality.

I tried to tackle the first problem by fully utilizing shifted window attention mechanism along with the skip-connection of U-Net to get rid of convolution. For the second one, I alternatively using self attention and cross attention operation, which integrates feature extraction and fusion to a single unified process. For the third problem I raise an anti patch merging operation to avoid noise.

The model shows a symmetric structure and performs better than many previous models across multiple metrics. 
![image](https://github.com/user-attachments/assets/25dd47fe-641d-49da-9d54-3c522b64397b)
![image](https://github.com/user-attachments/assets/37a23c1c-b129-4811-8b01-7b16c30f3ac2)
![image](https://github.com/user-attachments/assets/954432bd-9fd7-43b4-b4c7-098210da8674)
![image](https://github.com/user-attachments/assets/da8f6158-3b7c-4742-a86e-708d66c039f7)


