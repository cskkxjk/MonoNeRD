import numpy as np
import torch

import torch.nn as nn
import torch
from einops import rearrange

class Density(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, beta=0.1, beta_min=0.0001):
        super().__init__(beta=beta)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

def depth_to_disparity(depth):
    eps = 1e-6
    depth += eps
    disparity = 1. / depth
    return disparity

def disparity_to_depth(disparity):
    eps = 1e-6
    disparity += eps
    depth = 1. / disparity
    return depth

def construct_ray_warps(t_near=2, t_far=1e6, uniform=False):
    """Construct a bijection between metric distances and normalized distances.
        See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
        detailed explanation.
        Args:
        t_near: a tensor of near-plane distances.
        t_far: a tensor of far-plane distances.
        Returns:
        t_to_s: a function that maps distances to normalized distances in [0, 1].
        s_to_t: the inverse of t_to_s.
    """
    eps = 1e-6
    fn_fwd = lambda x: 1. / (x + eps)
    fn_inv = lambda x: 1. / (x + eps)
    s_near, s_far = [fn_fwd(x) for x in (t_near, t_far)]
    if uniform:
        t_to_s = lambda t: (t - t_near) / (t_far - t_near)
        s_to_t = lambda s: t_near + (t_far - t_near) * s
    else:
        t_to_s = lambda t: (fn_fwd(t) - s_near) / (s_far - s_near)
        s_to_t = lambda s: fn_inv(s * s_far + (1 - s) * s_near)
    return t_to_s, s_to_t

def sample_along_rays(batch_size, num_samples, randomized=False):
    """
    Sample disparity for frustum
    """
    # near > far in disparity
    # assert near > far
    t_vals = torch.linspace(0, 1., num_samples + 1, dtype=torch.float32)

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])

    return t_vals


def grid_generation(h, w):
    us = torch.linspace(0, w - 1, w, dtype=torch.float32)
    vs = torch.linspace(0, h - 1, h, dtype=torch.float32)
    vs, us = torch.meshgrid(vs, us)
    ones = torch.ones_like(vs)
    twos = 2 * ones
    # coord_2d = torch.stack([us, vs], dim=-1)
    norm_coord_2d = torch.stack([us / w, vs / h], dim=-1) * 2 - 1.
    coord_3d_depth1 = torch.stack([us, vs, ones], dim=-1)
    coord_3d_depth2 = torch.stack([us, vs, twos], dim=-1)
    coord_3d_depth = torch.cat((coord_3d_depth1[:, :, None, :], coord_3d_depth2[:, :, None, :]), dim=-2)
    return norm_coord_2d.float(), coord_3d_depth.float()

def compute_alpha_weights(density, sdist, dirs, s_to_t):
    '''
    :param density: # b c d h w
    :param sdist:
    :param dirs:
    :param s_to_t:
    :return:
    '''
    tdist = s_to_t(sdist)
    t_delta = tdist[..., 1:] - tdist[..., :-1]

    delta = t_delta[:, None, None, :] * torch.linalg.norm(dirs[..., None, :], dim=-1)
    density_delta = density.squeeze(1) * rearrange(delta, 'b h w d -> b d h w')

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat(
        [torch.zeros_like(density_delta[:, :1, :, :]), torch.cumsum(density_delta[:, :-1, :, :], dim=1)],
        dim=1))
    weights = alpha * trans

    return weights, tdist


def get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z):
    # largest index
    n_x = grid_res_x - 1
    n_y = grid_res_y - 1
    n_z = grid_res_z - 1

    # x-axis normal vectors
    X_1 = torch.cat(
        (grid[1:, :, :], (3 * grid[n_x, :, :] - 3 * grid[n_x - 1, :, :] + grid[n_x - 2, :, :]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid[1, :, :] + 3 * grid[0, :, :] + grid[2, :, :]).unsqueeze_(0), grid[:n_x, :, :]), 0)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size[0])

    # y-axis normal vectors
    Y_1 = torch.cat(
        (grid[:, 1:, :], (3 * grid[:, n_y, :] - 3 * grid[:, n_y - 1, :] + grid[:, n_y - 2, :]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid[:, 1, :] + 3 * grid[:, 0, :] + grid[:, 2, :]).unsqueeze_(1), grid[:, :n_y, :]), 1)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size[1])

    # z-axis normal vectors
    Z_1 = torch.cat(
        (grid[:, :, 1:], (3 * grid[:, :, n_z] - 3 * grid[:, :, n_z - 1] + grid[:, :, n_z - 2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid[:, :, 1] + 3 * grid[:, :, 0] + grid[:, :, 2]).unsqueeze_(2), grid[:, :, :n_z]), 2)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size[2])

    return [grid_normal_x, grid_normal_y, grid_normal_z]

def voxel_eikonal_loss(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z):
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y,
                                                                    grid_res_z)
    eikonal_loss = torch.mean(
        torch.abs(torch.pow(grid_normal_x[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) + \
                  torch.pow(grid_normal_y[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) + \
                  torch.pow(grid_normal_z[1:grid_res_x - 1, 1:grid_res_y - 1, 1:grid_res_z - 1], 2) - 1))
    return eikonal_loss

if __name__ == "__main__":
    # s_vals = sample_along_rays(1, 20, randomized=True)
    # t_to_s, s_to_t = construct_ray_warps(t_near=2, t_far=1e6)
    # t_vals = s_to_t(s_vals)
    grid = torch.randn(64, 64, 64)
    voxel_size = [4/64., 4 / 64., 4 /64.]
    grid_res_x = 64
    grid_res_y = 64
    grid_res_z = 64
    eikonal_loss = voxel_eikonal_loss(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z)
    print(1)