import octomap
import numpy as np
import open3d as o3d

import yaml
import json
from tqdm import tqdm

import utils
from range_image_map import segment_ground_multi_res
import evaluation
from batch_generation import get_batches


def get_scan_wise_labels(octree: octomap.OcTree,
                         scan: o3d.geometry.PointCloud(),
                         scan_no: int,
                         nearest_neighbors: int,
                         seq: int,
                         ground_removal: bool):
    """
       Provides final ground points calculated using certain
       resolution and after application of height filter

       Args:
           octree (octomap.OcTree): Octree created while generating static and dynamic map
           scan (o3d.geometry.PointCloud()): Point Cloud for which labels are required
           scan_no(int): Scan number for the particular scan
           nearest_neighbors(int): Number of nearest neighbors to be searched for labeling of unknown points
           seq(int): Sequence to which scans belong (Parameter to be used for KITTI Dataset)
           ground_removal(bool): True -> To apply ground removal
                                 False -> For results without ground removal
    """

    # Extracting labels from OcTree for input scan
    points = np.asarray(scan.points)
    labels = octree.getLabels(points)

    empty_idx, occupied_idx, unknown_idx = [], [], []

    for i in range(len(labels)):
        if labels[i] == 0:
            empty_idx.append(i)
        elif labels[i] == 1:
            occupied_idx.append(i)
        else:
            unknown_idx.append(i)

    colors = np.full((len(points), 3), [1.0, 0.0, 0.0])
    colors[empty_idx] = [0.0, 0.0, 1.0]

    scan.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    known_idx = np.concatenate((occupied_idx, empty_idx), axis=None)
    pcd_known = scan.select_by_index(known_idx)

    pred_tree = o3d.geometry.KDTreeFlann(pcd_known)
    color = np.asarray(pcd_known.colors)

    static_idx, dynamic_idx = [], []

    # Assigning labels to unknown labels
    for pt in unknown_idx:
        [_, idx, _] = pred_tree.search_knn_vector_3d(points[pt], nearest_neighbors)

        final_score = np.mean(color[idx, 0])

        if final_score > 0.5:
            static_idx.append(pt)
        else:
            dynamic_idx.append(pt)

    static_idx = np.concatenate((occupied_idx, static_idx), axis=None)
    dynamic_idx = np.concatenate((empty_idx, dynamic_idx), axis=None)
    static_idx = static_idx.astype(np.int32)
    dynamic_idx = dynamic_idx.astype(np.int32)
    labels = np.full((len(static_idx) + len(dynamic_idx),), 9)
    labels[dynamic_idx] = 251

    # Storing labels for input scan
    if ground_removal:
        file_name = f"./labels/{seq}/{str(scan_no).zfill(6)}"
    else:
        file_name = f"./labels/{seq}/orig/{str(scan_no).zfill(6)}"
    labels.reshape((-1)).astype(np.int32)
    labels.tofile(file_name + '.label')


def main():
    # Loading Parameters
    with open('config.yml') as f:
        config = yaml.safe_load(f)

    # Initializing Parameters
    seq = config['Dataset']['sequence']
    batch_size = config['Dataset']['batch_size']
    pose_file = config['Dataset']['path_to_poses']
    calib_file = config['Dataset']['path_to_calibration']
    path_to_scans = config['Dataset']['path_to_scans']
    path_to_gt = config['Dataset']['path_to_gt']

    nearest_neighbors = config['Octomap']['nearest_neighbors']
    resolution = config['Octomap']['resolution']
    ground_removal = config['Octomap']['ground_removal']
    height_filter = config['Octomap']['height_filter']
    height_threshold = 0.5
    octree = octomap.OcTree(resolution)

    hm_resolution = config['Height_Map']['resolution']
    fwd_range = (config['Height_Map']['backward_range'], config['Height_Map']['fwd_range'])
    side_range = (config['Height_Map']['right_range'], config['Height_Map']['left_range'])
    height_range = (config['Height_Map']['bottom'], config['Height_Map']['top'])

    store_pcd = config['Results']['store_pcd']
    store_individual_label = config['Results']['store_individual_label']

    poses = utils.load_poses(pose_file)
    inv_frame0 = np.linalg.inv(poses[0])

    # load calibrations
    T_cam_velo = utils.load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # convert kitti poses from camera coord to LiDAR coord
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # creating batches for the sequence
    batches, max_dis = get_batches(poses, batch_size)
    print(len(batches), "batches created")

    for batch in batches:
        octree.clear()
        octree = octomap.OcTree(resolution)
        start, end = batch[0], batch[1]
        pcds = {}
        final_map = o3d.geometry.PointCloud()
        start_idx, end_idx = start, end

        print(f"Running OctoMap from {batch[0]} till {batch[1]}")
        for scan_no in tqdm(range(start_idx, end_idx)):

            pcd = utils.convert_kitti_bin_to_pcd(f"{path_to_scans}{str(scan_no).zfill(6)}.bin")
            pcds[scan_no] = pcd

            # Applying ground segmentation on Point Cloud
            ground_pcd, non_ground_pcd, ground_indices, non_ground_indices = \
                segment_ground_multi_res(pcd=pcd,
                                         res=hm_resolution,
                                         fwd_range=fwd_range,
                                         side_range=side_range,
                                         height_range=height_range,
                                         height_threshold=-1.1)

            points = np.asarray(pcd.points)
            ground_pcd = ground_pcd.transform(poses[scan_no])
            ground_points = np.asarray(ground_pcd.points)

            ht_filter = np.where(points[:, 2] > height_threshold)[0]
            ht_points = points[ht_filter]

            final_map = final_map + pcd.transform(poses[scan_no])

            pose = poses[scan_no]

            # Inserting Point Cloud into OcTree
            octree.insertPointCloud(
                pointcloud=points,
                origin=np.array([pose[0][3], pose[1][3], pose[2][3]], dtype=float),
                maxrange=-1,
            )

            # Setting ground points as static in the Occupancy grid
            if ground_removal:
                for pt in range(len(ground_points)):
                    try:
                        key = octree.coordToKey(ground_points[pt])
                        node = octree.search(key)
                        node.setValue(200.0)
                    except Exception as e:
                        print(e)

            # Setting points above a certain height as static in the Occupancy grid
            if height_filter:
                for pt in range(len(ht_points)):
                    try:
                        key = octree.coordToKey(ht_points[pt])
                        node = octree.search(key)
                        node.setValue(200.0)
                    except Exception as e:
                        print(e)

            octree.updateInnerOccupancy()

        # Extracting labels from OcTree
        final_points = np.asarray(final_map.points)
        labels = octree.getLabels(final_points)

        occupied_idx, empty_idx, unknown_idx = [], [], []
        for i in range(len(labels)):
            if labels[i] == 1:
                occupied_idx.append(i)
            elif labels[i] == 0:
                empty_idx.append(i)
            else:
                unknown_idx.append(i)

        pcd_static = final_map.select_by_index(occupied_idx)
        pcd_dynamic = final_map.select_by_index(empty_idx)

        color_static = np.full((len(np.asarray(pcd_static.points)), 3), [1.0, 0.0, 0.0])
        color_dynamic = np.full((len(np.asarray(pcd_dynamic.points)), 3), [0.0, 0.0, 1.0])

        pcd_static.colors = o3d.utility.Vector3dVector(np.asarray(color_static))
        pcd_dynamic.colors = o3d.utility.Vector3dVector(np.asarray(color_dynamic))

        pcd = pcd_static + pcd_dynamic
        pred_tree = o3d.geometry.KDTreeFlann(pcd)
        color = np.asarray(pcd.colors)

        # Assigning labels to unknown labels
        for pt in unknown_idx:
            [_, idx, _] = pred_tree.search_knn_vector_3d(final_points[pt], nearest_neighbors)
            final_score = np.mean(color[idx, 0])

            if final_score > 0.5:
                occupied_idx.append(pt)
            else:
                empty_idx.append(pt)

        pcd_static = final_map.select_by_index(occupied_idx)
        pcd_dynamic = final_map.select_by_index(empty_idx)

        # Downsampling static and dynamic point cloud
        data = pcd_static.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                                      [max_dis, max_dis, max_dis])
        pcd_static = data[0]
        data = pcd_dynamic.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                                       [max_dis, max_dis, max_dis])

        pcd_dynamic = data[0]
        data = final_map.voxel_down_sample_and_trace(0.1, [-1 * max_dis, -1 * max_dis, -1 * max_dis],
                                                     [max_dis, max_dis, max_dis])
        down_idx = data[1]
        down_idx = down_idx[down_idx > 0]

        # Calculating static and dynamic indices for original map
        static_idx = list(set(occupied_idx) & set(down_idx))
        dynamic_idx = list(set(empty_idx) & set(down_idx))

        # Performing evaluation
        print("performing eval ...")
        ts, td, total_static, total_dynamic = evaluation.eval(static_idx=static_idx,
                                                              dynamic_idx=dynamic_idx,
                                                              start=start,
                                                              end=end,
                                                              poses=poses,
                                                              path_to_scan=path_to_scans,
                                                              path_to_gt=path_to_gt)
        print("TS", ts, "Total Static", total_static)
        print("TD", td, "Total Dynamic", total_dynamic)

        key = str(start) + "-" + str(end)
        res = {key: {"TS": ts, "TD": td, "Total Voxels": total_static, "Total Dynamic Points": total_dynamic,
                     "Accuracy": (ts + td) / (total_dynamic + total_static + 1e-8),
                     "Accuracy_1": (ts / (2 * total_static + 1e-8)) + (td / (2 * total_dynamic + 1e-8)),
                     "Recall": td / (total_dynamic + 1e-8)}}

        # Storing results
        json_data = json.dumps(res)
        if ground_removal:
            f = open(f'./json/{seq}.json', 'a+')
            f.write(json_data + "\n")
            f.close()
        else:
            f = open(f'./json/{seq}_orig.json', 'a+')
            f.write(json_data + "\n")
            f.close()

        # Storing static and dynamic point cloud for individual scans
        if store_pcd:
            o3d.io.write_point_cloud(f"./pcd/{seq}/static/{int(start)}-{int(end)}.pcd", pcd_static)
            o3d.io.write_point_cloud(f"./pcd/{seq}/dynamic/{int(start)}-{int(end)}.pcd", pcd_dynamic)

        # Storing labels for individual scans
        if store_individual_label:
            print("Storing Scan Wise Labels")
            for scan_no in tqdm(range(start, end)):
                pcd = pcds[scan_no]
                get_scan_wise_labels(octree, pcd, scan_no, nearest_neighbors, seq, ground_removal)


if __name__ == '__main__':
    main()
