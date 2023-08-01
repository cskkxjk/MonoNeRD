import pickle
import numpy as np
import os
from pathlib import Path
from mayavi import mlab
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
def draw(
    voxels,
    grid_coords,
    T_velo_2_cam,
    fov_mask,
    img_size,
    f,
    d=7,  # 7m - determine the size of the mesh representing the camera
    save_path=None,
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0]# - vox_origin[0]
    y = tri_points[:, 1]# - vox_origin[1]
    z = tri_points[:, 2]# - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]
    min = np.min(voxels)
    max = np.max(voxels)

    height_min = np.min(grid_coords[:, :, :, 2])
    height_max = np.max(grid_coords[:, :, :, 2])
    height_mask = grid_coords[:, :, :, 2] > 0.
    voxels[height_mask] = 0
    voxels[voxels < 1] = 0
    # voxels = voxels * 10

    # Attach the predicted class to every voxel
    grid_coords = np.concatenate([grid_coords, voxels[..., None]], axis=-1)

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the high and low density voxels inside FOV
    high_voxels = fov_grid_coords[
        (fov_grid_coords[..., 3] > 10)
    ]

    low_voxels = fov_grid_coords[
        (fov_grid_coords[..., 3] > 0) & (fov_grid_coords[..., 3] <= 10)
    ]

    figure = mlab.figure(size=(1248, 500), bgcolor=(1, 1, 1))
    scene = figure.scene
    # -------------------------------------------

    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Draw high occupied inside FOV voxels
    plt_plot_high = mlab.points3d(
        high_voxels[:, 0],
        high_voxels[:, 1],
        high_voxels[:, 2],
        high_voxels[:, 2],
        colormap="viridis",
        scale_factor=0.19,
        scale_mode='none',
        mode="cube",
        opacity=1.0,
        # vmin=0,
        # vmax=59,
    )

    # Draw low occupied inside FOV voxels
    plt_plot_low = mlab.points3d(
        low_voxels[:, 0],
        low_voxels[:, 1],
        low_voxels[:, 2],
        low_voxels[:, 2],
        colormap="viridis",
        scale_factor=0.19,
        scale_mode='none',
        mode="cube",
        opacity=1.0,
        # vmin=0,
        # vmax=59,
    )

    plt_plot_high.glyph.scale_mode = "scale_by_vector"
    plt_plot_low.glyph.scale_mode = "scale_by_vector"

    scene.x_minus_view()
    scene.camera.position = [-105.78471712503311, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [75.33599547459218, 212.05537696476563]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-82.23023730187387, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [52.01706044966453, 188.147579944259]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-62.76372505132905, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [32.74521332162516, 168.38907000995601]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-46.675698397986224, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [16.81806693481576, 152.05972295681303]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-33.37980860183513, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [3.6551360366261854, 138.56439481371967]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-22.391469927330093, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.12741123105909707, 127.41123105909706]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-13.310198295507746, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.11819374035279738, 118.19374035279738]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-5.805015128712419, 0.0, -1.2965779427546997]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.11057597943850014, 110.57597943850014]
    scene.camera.compute_view_plane_normal()
    # scene.render()
    scene.camera.position = [-5.654993888784655, 0.0, 1.9746148140270752]
    scene.camera.focal_point = [29.933952332217657, 0.0, -1.2965779427546997]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.09153014172437561, 0.0, 0.995802306261597]
    scene.camera.clipping_range = [0.11065350624202265, 110.65350624202264]
    scene.camera.compute_view_plane_normal()
    scene.render()
    # shot = mlab.screenshot()
    mlab.savefig(str(save_path))
    mlab.close()
    # mlab.show()

def main(file, save_2d_path=None, save_3d_path=None):
    scan = file
    with open(scan, "rb") as handle:
        b = pickle.load(handle)
    fov_mask_1 = b["fov_mask"]
    T_velo_2_cam = b["T_velo_2_cam"]
    f = b['f']
    density = b['density']
    bev = b['vis_bev']
    if save_2d_path is not None:
        bev_density = Image.fromarray(np.uint8(bev * 255.)).convert('RGB')
        bev_density.save(save_2d_path)
    coord = b['coord']
    draw(
        density,  # (256, 256, 32) np.uint16
        coord,
        T_velo_2_cam,  # (4, 4) np.float32
        # vox_origin,
        fov_mask_1,  # (n, ) bool n=256x256x32
        img_size=(1248, 320),
        f=f,
        d=2,
        save_path=save_3d_path,
    )


if __name__ == '__main__':
    root_path = Path('../../data/kitti/visualization/sequence_1')
    pkl_dir = root_path / 'vis_3d'
    assert pkl_dir.exists()
    save_3d_dir = root_path / 'density_3d'
    save_2d_dir = root_path / 'density_2d'
    save_3d_dir.mkdir(parents=True, exist_ok=True)
    save_2d_dir.mkdir(parents=True, exist_ok=True)
    for _, _, files in os.walk(pkl_dir):
        for pkl_file in tqdm(files):
            if pkl_file.endswith('.txt'):
                continue
            stem = pkl_file.split('.')[0]
            pkl_file = pkl_dir / pkl_file
            image_file = stem + '.png'
            save_3d_path = save_3d_dir / image_file
            save_2d_path = save_2d_dir / image_file
            main(pkl_file, save_2d_path, save_3d_path)