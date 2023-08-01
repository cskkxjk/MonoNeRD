import os
import numpy as np
from pathlib import Path
import cv2.cv2 as cv
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

def merge_images(image_2_path, density_2d_path, density_3d_path, merge_path):
    image_2 = cv.imread(str(image_2_path))
    density_2d = cv.imread(str(density_2d_path))
    density_3d = cv.imread(str(density_3d_path))
    image_2_shape = image_2.shape
    density_3d_shape = density_3d.shape
    new_img_w = 1000
    new_img_h = int(image_2_shape[0] * new_img_w / image_2_shape[1])
    new_density_3d_w = 1000
    new_density_3d_h = int(density_3d_shape[0] * new_density_3d_w / density_3d_shape[1])
    image_2 = cv.resize(image_2, dsize=(new_img_w, new_img_h))
    density_3d = cv.resize(density_3d, dsize=(new_density_3d_w, new_density_3d_h))[100:, :, :]
    merge_0 = np.concatenate((image_2, density_3d), axis=0)
    new_density_2d_h = merge_0.shape[0]
    new_density_2d_w = int(density_2d.shape[1] * new_density_2d_h / density_2d.shape[0])
    density_2d = cv.resize(density_2d, dsize=(new_density_2d_w, new_density_2d_h))
    merge_1 = np.concatenate((merge_0, density_2d), axis=1)
    # resolution need to be even number
    merge_1 = cv.resize(merge_1, dsize=(1600, 560))
    cv.imwrite(str(merge_path), merge_1)
    merge_1 = cv.cvtColor(merge_1, cv.COLOR_BGR2RGB)
    # cv.imshow('merge_1', merge_1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return merge_1


if __name__ == "__main__":
    sequence = 'sequence_0'
    root_path = Path('../../data/kitti/visualization')
    data_path = root_path / sequence
    image_2_dir = data_path / 'image_2'
    density_2d_image_dir = data_path / 'density_2d'
    density_3d_image_dir = data_path / 'density_3d'
    merge_image_dir = data_path / 'merge'
    merge_image_dir.mkdir(parents=True, exist_ok=True)
    video_clip = sequence + '.mp4'
    video_path = root_path / video_clip
    img_np_list = []
    for _, _, files in os.walk(image_2_dir):
        for filename in tqdm(files):
            assert filename.endswith('.png')
            image_2_path = image_2_dir / filename
            density_2d_path = density_2d_image_dir / filename
            density_3d_path = density_3d_image_dir / filename
            merge_path = merge_image_dir / filename
            merged_image = merge_images(image_2_path, density_2d_path, density_3d_path, merge_path)
            img_np_list.append(merged_image)
    rgb_video = ImageSequenceClip(img_np_list, fps=8)
    rgb_video.write_videofile(str(video_path),
                              # codec='mpeg4',
                              fps=8,
                              verbose=False,
                              logger=None)