import os
import numpy as np
from pathlib import Path
import shutil

def raw2object(raw_path, image_path, save_path, start_index, end_index):
    image_read_path = raw_path / image_path / 'image_02' / 'data'
    calib_cam2cam_path = raw_path / 'calib_cam_to_cam.txt'
    calib_imu2velo_path = raw_path / 'calib_imu_to_velo.txt'
    calib_velo2cam_path = raw_path / 'calib_velo_to_cam.txt'
    calib_save_path = save_path / 'calib'
    calib_save_path.mkdir(parents=True, exist_ok=True)
    image_save_path = save_path / 'image_2'
    image_save_path.mkdir(parents=True, exist_ok=True)
    calib_dict = {}
    valid_image_list = []
    with open(calib_cam2cam_path, encoding='utf-8') as f:
        text = f.readlines()
        calib_dict['P0'] = np.array(text[9].split(' ')[1:], dtype=np.float32)#.reshape(3, 4)
        calib_dict['P1'] = np.array(text[17].split(' ')[1:], dtype=np.float32)#.reshape(3, 4)
        calib_dict['P2'] = np.array(text[-9].split(' ')[1:], dtype=np.float32)#.reshape(3, 4)
        calib_dict['P3'] = np.array(text[-1].split(' ')[1:], dtype=np.float32)#.reshape(3, 4)
        calib_dict['R0_rect'] = np.array(text[8].split(' ')[1:], dtype=np.float32)

    with open(calib_velo2cam_path, encoding='utf-8') as f:
        text = f.readlines()
        R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
        T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

        calib_dict['Tr_velo_to_cam'] = np.concatenate([R, T], axis=1).reshape(-1,)

    with open(calib_imu2velo_path, encoding='utf-8') as f:
        text = f.readlines()
        R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
        T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

        calib_dict['Tr_imu_to_velo'] = np.concatenate([R, T], axis=1).reshape(-1,)
    for root, dirs, files in os.walk(image_read_path):
        for file in files:
            stem = file.split('.')[0]
            if int(stem) >= start_index and int(stem) <= end_index:
                calib_file = calib_save_path / f'{stem}.txt'
                image_src_path = image_read_path / f'{stem}.png'
                image_target_path = image_save_path / f'{stem}.png'
                shutil.copy(image_src_path, image_target_path)
                with open(calib_file, 'w') as f:
                    print('P0: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["P0"][0], calib_dict["P0"][1], calib_dict["P0"][2], calib_dict["P0"][3],
                           calib_dict["P0"][4], calib_dict["P0"][5], calib_dict["P0"][6], calib_dict["P0"][7],
                           calib_dict["P0"][8], calib_dict["P0"][9], calib_dict["P0"][10], calib_dict["P0"][11]), file=f)
                    print('P1: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["P1"][0], calib_dict["P1"][1], calib_dict["P1"][2], calib_dict["P1"][3],
                           calib_dict["P1"][4], calib_dict["P1"][5], calib_dict["P1"][6], calib_dict["P1"][7],
                           calib_dict["P1"][8], calib_dict["P1"][9], calib_dict["P1"][10], calib_dict["P1"][11]),
                          file=f)
                    print('P2: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["P2"][0], calib_dict["P2"][1], calib_dict["P2"][2], calib_dict["P2"][3],
                           calib_dict["P2"][4], calib_dict["P2"][5], calib_dict["P2"][6], calib_dict["P2"][7],
                           calib_dict["P2"][8], calib_dict["P2"][9], calib_dict["P2"][10], calib_dict["P2"][11]),
                          file=f)
                    print('P3: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["P3"][0], calib_dict["P3"][1], calib_dict["P3"][2], calib_dict["P3"][3],
                           calib_dict["P3"][4], calib_dict["P3"][5], calib_dict["P3"][6], calib_dict["P3"][7],
                           calib_dict["P3"][8], calib_dict["P3"][9], calib_dict["P3"][10], calib_dict["P3"][11]),
                          file=f)
                    print('R0_rect: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["R0_rect"][0], calib_dict["R0_rect"][1], calib_dict["R0_rect"][2], calib_dict["R0_rect"][3],
                           calib_dict["R0_rect"][4], calib_dict["R0_rect"][5], calib_dict["R0_rect"][6], calib_dict["R0_rect"][7],
                           calib_dict["R0_rect"][8]),
                          file=f)
                    print('Tr_velo_to_cam: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["Tr_velo_to_cam"][0], calib_dict["Tr_velo_to_cam"][1], calib_dict["Tr_velo_to_cam"][2],
                           calib_dict["Tr_velo_to_cam"][3],
                           calib_dict["Tr_velo_to_cam"][4], calib_dict["Tr_velo_to_cam"][5], calib_dict["Tr_velo_to_cam"][6],
                           calib_dict["Tr_velo_to_cam"][7],
                           calib_dict["Tr_velo_to_cam"][8], calib_dict["Tr_velo_to_cam"][9], calib_dict["Tr_velo_to_cam"][10],
                           calib_dict["Tr_velo_to_cam"][11]),
                          file=f)
                    print('Tr_imu_to_velo: %.6e  %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e' %
                          (calib_dict["Tr_imu_to_velo"][0], calib_dict["Tr_imu_to_velo"][1],
                           calib_dict["Tr_imu_to_velo"][2],
                           calib_dict["Tr_imu_to_velo"][3],
                           calib_dict["Tr_imu_to_velo"][4], calib_dict["Tr_imu_to_velo"][5],
                           calib_dict["Tr_imu_to_velo"][6],
                           calib_dict["Tr_imu_to_velo"][7],
                           calib_dict["Tr_imu_to_velo"][8], calib_dict["Tr_imu_to_velo"][9],
                           calib_dict["Tr_imu_to_velo"][10],
                           calib_dict["Tr_imu_to_velo"][11]),
                          file=f)


def parse_calib(mode, calib_path=None):
    if mode == '3d':
        with open(calib_path, encoding='utf-8') as f:
            text = f.readlines()
            P0 = np.array(text[0].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P1 = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(text[3].split(' ')[1:], dtype=np.float32).reshape(3, 4)

            Tr_velo_to_cam = np.array(text[5].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            Tr_imu_to_velo = np.array(text[6].split(' ')[1:], dtype=np.float32).reshape(3, 4)

            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[4].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1

            Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
            '''lidar to image pixel plane'''
            l2p = np.dot(np.dot(P2, R_rect), Tr_velo_to_cam)
            '''lidar to image plane'''
            l2i = np.dot(R_rect_tmp, Tr_velo_to_cam[:3])

    elif mode == 'raw':
        calib_cam2cam_path, velo2cam_calib_path = calib_path
        with open(velo2cam_calib_path, encoding='utf-8') as f:
            text = f.readlines()
            R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

            trans = np.concatenate([R, T], axis=1)
            vel2cam = trans.copy()

            Tr_velo_to_cam = np.concatenate([trans, np.array([[0, 0, 0, 1]])], axis=0)

        with open(calib_cam2cam_path, encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[-9].split(' ')[1:], dtype=np.float32).reshape(3, 4)
            R_rect = np.zeros((4, 4))
            R_rect_tmp = np.array(text[8].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            R_rect[:3, :3] = R_rect_tmp
            R_rect[3, 3] = 1

            '''lidar to image pixel plane'''
            l2p = np.dot(np.dot(P2, R_rect), Tr_velo_to_cam)
            '''lidar to image plane'''
            l2i = np.dot(R_rect_tmp, vel2cam)

    calib = {
        'P2': P2,
        'l2p': l2p,
        'l2i': l2i
    }
    return calib


if __name__ == "__main__":
    raw_path = Path('/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/benchmarks/KITTI_RAW_DATA/raw_data/2011_10_03')
    image_path = Path('2011_10_03_drive_0042_sync')
    save_path = Path('/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/xujunkai/kitti3d/visualization/sequence_1')
    start_index = 180
    end_index = 210
    raw2object(raw_path, image_path, save_path, start_index, end_index)
    print(1)