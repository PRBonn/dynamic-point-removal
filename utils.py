import open3d as o3d
import numpy as np
from time_utils import timeit

def load_label(label_path):
	label = np.fromfile(label_path, dtype=np.int32)
	label = label.reshape((-1))
  
	return label

def convert_kitti_bin_to_pcd(binFilePath):
	bin_pcd = np.fromfile(binFilePath, dtype=np.float32)
	#print(type(bin_pcd))
	points = bin_pcd.reshape((-1, 4))[:, 0:3]
	o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	return o3d_pcd


""" Using dpeth as a filter to label points as static or dynamic
	Currently using 40m as a threshold
"""

def depth_filtering(gt_pcd, pred_label, thresh=40):
	gt_points = np.asarray(gt_pcd.points)

	for i in range(len(gt_points)):
		dist = np.sqrt(gt_points[i][0]**2 + gt_points[i][1]**2 + gt_points[i][2]**2)
		if dist > thresh:
			pred_label[i] = 9

	return pred_label


""" Using height as a filter to label points as static or dynamic
	Currently using 0.3m as a threshold
"""

def height_filtering(gt_pcd, pred_label, thresh=0.3):
	gt_points = np.asarray(gt_pcd.points)

	pred_label[gt_points[:, 2] > thresh] = 9

	return pred_label
	


# Function to calculate the metrics for each scan

def get_indices(gt_label, pred_label):
	tp_idx, tn_idx, fp_idx, fn_idx = [], [], [], []

	for i in range(len(pred_label)):
		if pred_label[i] == gt_label[i] == 9:
			tp_idx.append(i)
		elif pred_label[i] == gt_label[i] == 251:
			tn_idx.append(i)
		elif pred_label[i] == 9 and gt_label[i] == 251:
			fp_idx.append(i)
		elif pred_label[i] == 251 and gt_label[i] == 9:
			fn_idx.append(i)

	return len(tp_idx), len(fp_idx), len(tn_idx), len(fn_idx)



def create_range_image(points, labels, type_, seq, proj_fov_up=3.0, proj_fov_down=-25.0, proj_W=1024, proj_H=64):

	proj_range = np.full((proj_H, proj_W), -1,
							  dtype=np.float32)

	# unprojected range (list of depths for each point)
	unproj_range = np.zeros((0, 1), dtype=np.float32)

	# projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
	proj_xyz = np.full((proj_H, proj_W, 3), -1,
							dtype=np.float32)

	# projected remission - [H,W] intensity (-1 is no data)
	proj_remission = np.full((proj_H, proj_W), -1,
								  dtype=np.float32)

	proj_labels =  np.full((proj_H, proj_W, 3), -1,
								  dtype=np.float32)

	# projected index (for each pixel, what I am in the pointcloud)
	# [H,W] index (-1 is no data)
	proj_idx = np.full((proj_H, proj_W), -1,
							dtype=np.int32)

	# for each point, where it is in the range image
	proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
	proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

	# mask containing for each pixel, if it contains a point or not
	proj_mask = np.zeros((proj_H, proj_W),
							  dtype=np.int32)

	fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
	fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
	fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

	# get depth of all points
	depth = np.linalg.norm(points, 2, axis=1)

	# get scan components
	scan_x = points[:, 0]
	scan_y = points[:, 1]
	scan_z = points[:, 2]

	# get angles of all points
	yaw = -np.arctan2(scan_y, scan_x)
	pitch = np.arcsin(scan_z / depth)

	# get projections in image coords
	proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
	proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

	# scale to image size using angular resolution
	proj_x *= proj_W                              # in [0.0, W]
	proj_y *= proj_H                              # in [0.0, H]

	# round and clamp for use as index
	proj_x = np.floor(proj_x)
	proj_x = np.minimum(proj_W - 1, proj_x)
	proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1] 

	proj_y = np.floor(proj_y)
	proj_y = np.minimum(proj_H - 1, proj_y)
	proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]


	# copy of depth in original order
	unproj_range = np.copy(depth)

	# order in decreasing depth
	indices = np.arange(depth.shape[0])
	order = np.argsort(depth)[::-1]
	depth = depth[order]
	indices = indices[order]
	points = points[order]
	proj_y = proj_y[order]
	proj_x = proj_x[order]
	labels = labels[order]

	p2c = {}
	for k, v in zip(points, zip(proj_y, proj_x)):
		p2c[k.tobytes()] = v

	label_mapping = {}
	for k, v in zip(zip(proj_y, proj_x), labels):
		label_mapping[k] = v
	# assing to images

	int_label = []
	for i in range(len(labels)):
		int_label.append([int(l) for l in labels[i]])

	proj_int_labels =  np.full((proj_H, proj_W, 3), -1,
								  dtype=np.uint8)

	proj_range[proj_y, proj_x] = depth
	proj_xyz[proj_y, proj_x] = points
	proj_idx[proj_y, proj_x] = indices
	proj_labels[proj_y, proj_x] = labels
	proj_int_labels[proj_y, proj_x] = int_label
	proj_mask = (proj_idx > 0).astype(np.float32)


	return proj_xyz, proj_labels, p2c, label_mapping


def load_calib(calib_path):
	T_cam_velo = []
	try:
		with open(calib_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				if 'Tr:' in line:
					line = line.replace('Tr:', '')
					T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
					T_cam_velo = T_cam_velo.reshape(3, 4)
					T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
  
	except FileNotFoundError:
		print('Calibrations are not avaialble.')
  
	return np.array(T_cam_velo)


def load_poses(pose_path):

	# Read and parse the poses
	poses = []
	try:
		if '.txt' in pose_path:
			with open(pose_path, 'r') as f:
				lines = f.readlines()
				for line in lines:
					T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
					T_w_cam0 = T_w_cam0.reshape(3, 4)
					T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
					poses.append(T_w_cam0)
		else:
			poses = np.load(pose_path)['arr_0']
	except FileNotFoundError:
		print('Ground truth poses are not avaialble.')

	return np.array(poses)


def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


