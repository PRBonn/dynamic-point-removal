import open3d as o3d
import numpy as np
from tqdm import tqdm
import utils


def get_static_dynamic_idx(gt_map):
    colors = np.asarray(gt_map.colors)
    static_idx = np.where(colors[:, 0] == 1.0)[0]
    dynamic_idx = np.where(colors[:, 2] == 1.0)[0]

    return static_idx, dynamic_idx


def eval(static_idx: np.ndarray,
         dynamic_idx: np.ndarray,
         start: int,
         end: int,
         poses: np.ndarray,
         path_to_scan: str,
         path_to_gt: str,
         ):
    """
          Creates smaller batches for larger sequences

          Args:
              static_idx (np.ndarray): List of indices corresponding to the static points in Point cloud
              dynamic_idx (np.ndarray): List of indices corresponding to the dynamic points in Point cloud
              start (int): Starting index of batch
              end (int): Ending index of batch
              poses (List: nx4x4): Poses for the LiDAR scans
              path_to_scan (str): Path to the scans
              path_to_gt (str): Path to ground truth labels

          Returns:
              ts: Total number of correctly identified static points
              td: Total number of correctly identified dynamic points
              total_static: Total number of static points in ground truth
              total_dynamic: Total number of dynamic points in ground truth

   """
    gt_map = o3d.geometry.PointCloud()

    for scan_no in tqdm(range(start, end)):
        pcd = utils.convert_kitti_bin_to_pcd(f"{path_to_scan}{str(scan_no).zfill(6)}.bin")

        # Loading ground truth labels
        gt_label_file = f"{path_to_gt}{str(scan_no).zfill(6)}.label"
        gt_label = utils.load_label(gt_label_file)

        gt_label = gt_label & 0xFFFF
        mask = gt_label > 200
        gt_label[mask] = 251
        gt_label[~mask] = 9

        colors = np.full((len(np.asarray(pcd.points)), 3), [1.0, 0.0, 0.0])
        colors[gt_label == 251] = [0.0, 0.0, 1.0]

        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

        # Creating ground truth map
        gt_map = gt_map + pcd.transform(poses[scan_no])

    gt_static_idx, gt_dynamic_idx = get_static_dynamic_idx(gt_map)

    max_dis = 100000

    # Downsampling created ground truth map
    data = gt_map.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                              [max_dis, max_dis, max_dis])
    down_idx = data[1]

    # Calculating result statistics
    down_idx = down_idx[down_idx > 0]
    gt_static_idx_ds = list(set(down_idx).intersection(gt_static_idx))
    gt_dynamic_idx_ds = list(set(down_idx).intersection(gt_dynamic_idx))

    true_static = list(set(gt_static_idx_ds) & set(list(static_idx)))
    true_dynamic = list(set(gt_dynamic_idx_ds) & set(list(dynamic_idx)))

    total_static = len(gt_static_idx_ds)
    total_dynamic = len(gt_dynamic_idx_ds)

    ts, td = len(true_static), len(true_dynamic)

    return ts, td, total_static, total_dynamic
