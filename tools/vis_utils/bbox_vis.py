import numpy as np
from pathlib import Path
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from skimage import io

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects


def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.truncation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.loc, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.loc[0], self.loc[1], self.loc[2],
                       self.ry)
        return kitti_str

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    calib_data = {}
    for key in ['P2', 'P3', 'R0_rect', 'Tr_velo_to_cam']:
        for line in lines:
            line = line.strip()
            splits = [x for x in line.split(' ') if len(x.strip()) > 0]
            if splits[0][:-1] == key:
                obj = splits[1:]
                calib_data[key] = np.array(obj, dtype=np.float32)
                break

    return {'P2': calib_data['P2'].reshape(3, 4),
            'P3': calib_data['P3'].reshape(3, 4),
            'R0': calib_data['R0_rect'].reshape(3, 3),
            'Tr_velo2cam': calib_data['Tr_velo_to_cam'].reshape(3, 4)}

class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.P3 = calib['P3']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4
        self.flipped = False
        self.offsets = [0, 0]

    @property
    def cu(self):
        return self.P2[0, 2]

    @property
    def cv(self):
        return self.P2[1, 2]

    @property
    def fu(self):
        return self.P2[0, 0]

    @property
    def fv(self):
        return self.P2[1, 1]

    @property
    def tx(self):
        return self.P2[0, 3] / (-self.fu)

    @property
    def ty(self):
        return self.P2[1, 3] / (-self.fv)

    @property
    def txyz(self):
        return np.matmul(np.linalg.inv(self.P2[:3, :3]), self.P2[:3, 3:4]).squeeze(-1)

    @property
    def K(self):
        return self.P2[:3, :3]

    @property
    def K3x4(self):
        return np.concatenate([self.P2[:3, :3], np.zeros_like(self.P2[:3, 3:4])], axis=1)

    @property
    def inv_K(self):
        return np.linalg.inv(self.K)

    def global_scale(self, scale_factor):
        self.P2[:, 3:4] *= scale_factor
        self.P3[:, 3:4] *= scale_factor

    def scale(self, scale_factor):
        self.P2[:2, :] *= scale_factor
        self.P3[:2, :] *= scale_factor

    def offset(self, offset_x, offset_y):
        K = self.K.copy()
        inv_K = self.inv_K
        T2 = np.matmul(inv_K, self.P2)
        T3 = np.matmul(inv_K, self.P3)
        K[0, 2] -= offset_x
        K[1, 2] -= offset_y
        self.P2 = np.matmul(K, T2)
        self.P3 = np.matmul(K, T3)
        self.offsets[0] += offset_x
        self.offsets[1] += offset_y

    def fliplr(self, image_width):
        # mirror using y-z plane of cam 0
        assert not self.flipped

        K = self.P2[:3, :3].copy()
        inv_K = np.linalg.inv(K)
        T2 = np.matmul(inv_K, self.P2)
        # T3 = np.matmul(inv_K, self.P3)
        T2[0, 3] *= -1
        # T3[0, 3] *= -1

        K[0, 2] = image_width - 1 - K[0, 2]
        # self.P3 = np.matmul(K, T2)
        # self.P2 = np.matmul(K, T3)

        self.P2 = np.matmul(K, T2)
        # delete useless parameters to avoid bugs
        del self.R0, self.V2C

        self.flipped = not self.flipped

    @property
    def fu_mul_baseline(self):
        return np.abs(self.P2[0, 3] - self.P3[0, 3])

    @staticmethod
    def cart_to_hom(pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack(
            (pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        if self.flipped:
            raise NotImplementedError

        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack(
            (self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack(
            (R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack(
            (self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(
            np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    @staticmethod
    def rect_to_lidar_pseudo(pts_rect):
        pts_rect_hom = Calibration.cart_to_hom(pts_rect)
        T = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        if self.flipped:
            raise NotImplementedError
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        # pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        pts_rect = np.dot(pts_lidar_hom, self.V2C.T)
        pts_rect = np.dot(pts_rect, self.R0.T)
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    @staticmethod
    def lidar_pseudo_to_rect(pts_lidar):
        pts_lidar_hom = Calibration.cart_to_hom(pts_lidar)
        T = np.array([[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
        pts_rect = np.dot(pts_lidar_hom, T)
        return pts_rect

    def torch_lidar_pseudo_to_rect(self, pts_lidar):
        pts_lidar_hom = torch.cat([pts_lidar, torch.ones_like(pts_lidar[..., -1:])], dim=-1)
        T = np.array([[0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
        T = torch.from_numpy(T).cuda()
        pts_rect = torch.matmul(pts_lidar_hom, T)
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def rect_to_right_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P3.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - \
            self.P3.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def torch_rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = torch.cat([pts_rect, torch.ones_like(pts_rect[..., -1:])], dim=-1)
        pts_2d_hom = torch.matmul(pts_rect_hom, torch.from_numpy(self.P2.T).cuda())
        pts_img = pts_2d_hom[..., 0:2] / pts_rect_hom[..., 2:3]
        return pts_img

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        if self.flipped:
            raise NotImplementedError
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate(
            (corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :,
                                          2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate(
            (x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate(
            (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner



def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def vis_lidar_and_bbox(index, image_2_dir, velodyne_dir, calib_dir, gt_label_dir, pred_dir, baseline_dir):
    # get image
    image_2_file = image_2_dir / ('%s.png' % index)
    plt.imshow(io.imread(image_2_file))
    plt.show()
    # get lidar point cloud
    velodyne_file = velodyne_dir / ('%s.bin' % index)
    raw_point = np.fromfile(str(velodyne_file), dtype=np.float32).reshape(-1, 4)[:, 0:3]

    # get calib
    calib_file = calib_dir / ('%s.txt' % index)
    calib = Calibration(calib_file)

    # get gt label
    label_file = gt_label_dir / ('%s.txt' % index)
    obj_list = get_objects_from_label(label_file)
    gt = {}
    gt['name'] = np.array([obj.cls_type for obj in obj_list])
    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(gt['name'])
    gt_index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    gt['index'] = np.array(gt_index, dtype=np.int32)
    gt['3dbbox'] = [obj.generate_corners3d() for obj in obj_list][:num_objects]
    gt['gt_boxes_lidar'] = []
    for bbox in gt['3dbbox']:
        gt_box_lidar = calib.rect_to_lidar(bbox)
        gt['gt_boxes_lidar'].append(gt_box_lidar)

    # get pred label
    pred_file = pred_dir / ('%s.txt' % index)
    pred_obj_list = get_objects_from_label(pred_file)
    pred = {}
    pred['name'] = np.array([obj.cls_type for obj in pred_obj_list])
    num_objects = len([obj.cls_type for obj in pred_obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(pred['name'])
    pred_index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    pred['index'] = np.array(pred_index, dtype=np.int32)
    pred['3dbbox'] = [obj.generate_corners3d() for obj in pred_obj_list][:num_objects]
    pred['confidence'] = [obj.score for obj in pred_obj_list][:num_objects]
    pred['pred_boxes_lidar'] = []
    for bbox in pred['3dbbox']:
        pred_box_lidar = calib.rect_to_lidar(bbox)
        pred['pred_boxes_lidar'].append(pred_box_lidar)

    # get baseline label
    bs_file = baseline_dir / ('%s.txt' % index)
    bs_obj_list = get_objects_from_label(bs_file)
    bs = {}
    bs['name'] = np.array([obj.cls_type for obj in bs_obj_list])
    num_objects = len([obj.cls_type for obj in bs_obj_list if obj.cls_type != 'DontCare'])
    num_gt = len(bs['name'])
    bs_index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    bs['index'] = np.array(bs_index, dtype=np.int32)
    bs['3dbbox'] = [obj.generate_corners3d() for obj in bs_obj_list][:num_objects]
    bs['confidence'] = [obj.score for obj in bs_obj_list][:num_objects]
    bs['bs_boxes_lidar'] = []
    for bbox in bs['3dbbox']:
        bs_box_lidar = calib.rect_to_lidar(bbox)
        bs['bs_boxes_lidar'].append(bs_box_lidar)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='kitti')
    opt = vis.get_render_option()
    opt.point_size = 1
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)

    pcd.paint_uniform_color([1, 1, 1])
    vis.add_geometry(pcd)

    for bbox in gt['gt_boxes_lidar']:
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
        line_set = o3d.geometry.LineSet()
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.points = o3d.utility.Vector3dVector(bbox)
        vis.add_geometry(line_set)

    for confidence, bbox in zip(pred['confidence'], pred['pred_boxes_lidar']):
        if confidence < 0.25:
            continue
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
        line_set = o3d.geometry.LineSet()
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.points = o3d.utility.Vector3dVector(bbox)
        vis.add_geometry(line_set)

    for confidence, bbox in zip(bs['confidence'], bs['bs_boxes_lidar']):
        # if confidence < 0.25:
        #     continue
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        colors = np.array([[0, 0, 1] for j in range(len(lines_box))])
        line_set = o3d.geometry.LineSet()
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.points = o3d.utility.Vector3dVector(bbox)
        vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    root_path = Path('E:\\personal\\xujunkai\\kitti3d')
    val_file_txt = root_path / 'ImageSets' / 'val.txt'
    image_2_dir = root_path / 'training' / 'image_2'
    calib_dir = root_path /'training' / 'calib'
    gt_label_dir = root_path / 'training' / 'label_2'
    velodyne_dir = root_path / 'training' / 'velodyne'
    pred_dir = Path('F:\\MonoNeRD\\outputs\\configs_stereo_kitti_models\\'
                    'mononerd.3d-and-bev.qkv-left-rgbd-sdf-right-rgb\\'
                    'eval\\eval_with_train\\epoch_51\\val\\final_result\\data')
    baseline_dir = Path('F:\\MonoNeRD\\outputs\\configs_stereo_kitti_models\\'
                    'mononerd.3d-and-bev.qvk-naive\\'
                    'eval\\eval_with_train\\epoch_52\\val\\final_result\\data')
    val_list = []
    with open(val_file_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        val_list.append(line[:-1])
        index = line[:-1]
        print(index)
        vis_lidar_and_bbox(index, image_2_dir, velodyne_dir, calib_dir, gt_label_dir, pred_dir, baseline_dir)
    # index = '000005'
    # assert index in val_list
    # # get image
    # image_2_file = image_2_dir / ('%s.png' % index)
    # plt.imshow(io.imread(image_2_file))
    # plt.show()
    # # get lidar point cloud
    # velodyne_file = velodyne_dir / ('%s.bin' % index)
    # raw_point = np.fromfile(str(velodyne_file), dtype=np.float32).reshape(-1, 4)[:, 0:3]
    #
    # # get calib
    # calib_file = calib_dir / ('%s.txt' % index)
    # calib = Calibration(calib_file)
    #
    # # get gt label
    # label_file = gt_label_dir / ('%s.txt' % index)
    # obj_list = get_objects_from_label(label_file)
    # gt = {}
    # gt['name'] = np.array([obj.cls_type for obj in obj_list])
    # num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
    # num_gt = len(gt['name'])
    # gt_index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    # gt['index'] = np.array(gt_index, dtype=np.int32)
    # gt['3dbbox'] = [obj.generate_corners3d() for obj in obj_list][:num_objects]
    # gt['gt_boxes_lidar'] = []
    # for bbox in gt['3dbbox']:
    #     gt_box_lidar = calib.rect_to_lidar(bbox)
    #     gt['gt_boxes_lidar'].append(gt_box_lidar)
    #
    # # get pred label
    # pred_file = pred_dir / ('%s.txt' % index)
    # pred_obj_list = get_objects_from_label(pred_file)
    # pred = {}
    # pred['name'] = np.array([obj.cls_type for obj in pred_obj_list])
    # num_objects = len([obj.cls_type for obj in pred_obj_list if obj.cls_type != 'DontCare'])
    # num_gt = len(pred['name'])
    # pred_index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    # pred['index'] = np.array(pred_index, dtype=np.int32)
    # pred['3dbbox'] = [obj.generate_corners3d() for obj in pred_obj_list][:num_objects]
    # pred['pred_boxes_lidar'] = []
    # for bbox in pred['3dbbox']:
    #     pred_box_lidar = calib.rect_to_lidar(bbox)
    #     pred['pred_boxes_lidar'].append(pred_box_lidar)
    #
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name='kitti')
    # opt = vis.get_render_option()
    # opt.point_size = 1
    # opt.background_color = np.asarray([0, 0, 0])
    #
    # pcd = o3d.open3d.geometry.PointCloud()
    # pcd.points = o3d.open3d.utility.Vector3dVector(raw_point)
    #
    # pcd.paint_uniform_color([1, 1, 1])
    # vis.add_geometry(pcd)
    #
    # for bbox in gt['gt_boxes_lidar']:
    #     lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
    #                           [0, 4], [1, 5], [2, 6], [3, 7]])
    #     colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
    #     line_set = o3d.geometry.LineSet()
    #     line_set.lines = o3d.utility.Vector2iVector(lines_box)
    #     line_set.colors = o3d.utility.Vector3dVector(colors)
    #     line_set.points = o3d.utility.Vector3dVector(bbox)
    #     vis.add_geometry(line_set)
    #
    # for bbox in pred['pred_boxes_lidar']:
    #     lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
    #                           [0, 4], [1, 5], [2, 6], [3, 7]])
    #     colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
    #     line_set = o3d.geometry.LineSet()
    #     line_set.lines = o3d.utility.Vector2iVector(lines_box)
    #     line_set.colors = o3d.utility.Vector3dVector(colors)
    #     line_set.points = o3d.utility.Vector3dVector(bbox)
    #     vis.add_geometry(line_set)
    # vis.run()
    # vis.destroy_window()
