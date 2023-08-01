import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().detach().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

if __name__ == "__main__":
    density_bar = torch.linspace(100, 1, 100)[:, None].repeat(1, 20)
    density_bar = visualize_depth(density_bar)
    plt.imshow(density_bar.permute(1,2,0))
    plt.yticks([0, 100], ['high', 'low'])
    plt.xticks([10], ['density'])
    # plt.xlabel('SDF')
    # plt.ylabel('Density')
    plt.show()
    print(1)