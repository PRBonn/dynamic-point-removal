import cv2
import numpy as np
import open3d as o3d


def point_cloud_2_birdseye(points,
                           res=0.05,
                           side_range=(-50., 50.),
                           fwd_range=(-50., 50.),
                           height_range=(-2., 1.5)):
    """
        Creates HeightMap from the given set of points.

        Args:
            points (numpy array nx3): Points for which a range image is required to be generated
            res (float): Resolution to be used for generating range image.
            fwd_range(tuple(float, float)): x_value for which points are to be considered.
            side_range(tuple(float, float)): y_value for which points are to be considered.
            height_range(tuple(float, float)): z_value for which points are to be considered.

        Returns:
            im: HeightMap
            p2c: Mapping of points from HeightMap to Point Cloud.
    """

    # Separating out x, y and z coordinates
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # Applying horizontal, and vertical filters
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # Extracting points following the filter above created
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    points = points[indices]

    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # Clipping values according to the height range specified
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # Creating a pixel to point mapping
    p2c = {}
    for k, v in zip(points, zip(y_img, x_img)):
        p2c[k.tobytes()] = v

    # Scaling pixel values from 0-255
    pixel_values = (((pixel_values - height_range[0]) /
                     float(height_range[1] - height_range[0])) * 255).astype(np.uint8)

    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    im[y_img, x_img] = pixel_values

    return im, p2c


def segment_ground_indices_ht(pcd,
                              res,
                              fwd_range,
                              side_range,
                              height_range,
                              height_threshold):

    """
    Provides probable ground points calculated using certain
    resolution and after application of height filter

    Args:
        pcd (o3d.geometry.PointCloud()): Point cloud for which ground points are required
        res (float): Resolution to be used for generating range image.
        fwd_range(tuple(float, float)): x_value for which points are to be considered.
        side_range(tuple(float, float)): y_value for which points are to be considered.
        height_range(tuple(float, float)): z_value for which points are to be considered.
        height_threshold(float): Height threshold for correcting wrongly classified non-ground points

    Returns:
        ground_pcd: Point Cloud consisting of only ground points
        non_ground_pcd: Point Cloud consisting of non-ground points
        ground_indices: Indices of points that belong to ground from original Point Cloud
        non_ground_indices: Indices of points that belong to non-ground from original Point Cloud

    """

    # Creating HeightMap from point cloud
    points = np.asarray(pcd.points)
    img, p2c = point_cloud_2_birdseye(points, res=res, side_range=side_range, fwd_range=fwd_range,
                                      height_range=height_range)
    img = np.array(img)

    # Applying Canny Edge Detector
    CANNY_THRESH_1 = 15
    CANNY_THRESH_2 = 300

    edges = cv2.Canny(img, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)

    colors = np.full((len(points), 3), [1.0, 0.0, 0.0])
    ground_indices = []

    # Calculating ground indices for input scan
    for i in range(len(points)):
        if points[i].tobytes() in p2c.keys():
            y, x = p2c[points[i].tobytes()]
            if edges[y, x] == 0:
                ground_indices.append(i)

    colors[ground_indices] = [0.0, 0.0, 0.0]
    colors = np.asarray(colors)

    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    ground_indices = []
    non_ground_indices = []

    for i in range(len(colors)):
        if list(colors[i]) == [0.0, 0.0, 0.0]:
            ground_indices.append(i)
        else:
            non_ground_indices.append(i)

    ground_points = points[ground_indices]

    # Applying height threshold on ground points detected above
    ht_filter = np.where(ground_points[:, 2] > height_threshold)[0]

    # Reverting non-ground points back from ground points
    non_ground_ext = np.asarray(ground_indices)[ht_filter]
    non_ground_indices.extend(non_ground_ext)

    ground_indices = np.delete(ground_indices, ht_filter)

    ground_pcd = pcd.select_by_index(ground_indices)
    non_ground_pcd = pcd.select_by_index(ground_indices, invert=True)

    return ground_pcd, non_ground_pcd, ground_indices, non_ground_indices


def segment_ground_multi_res(pcd,
                             res,
                             fwd_range,
                             side_range,
                             height_range,
                             height_threshold):
    """
        Provides final ground points calculated using certain
        resolution and after application of height filter

        Args:
            pcd (o3d.geometry.PointCloud()): Point cloud for which ground points are required
            res (List: float): List of resolutions to be used for generating range image.
            fwd_range(tuple(float, float)): x_value for which points are to be considered.
            side_range(tuple(float, float)): y_value for which points are to be considered.
            height_range(tuple(float, float)): z_value for which points are to be considered.
            height_threshold(float): Height threshold for correcting wrongly classified non-ground points

        Returns:
            ground_pcd: Point Cloud consisting of only ground points
            non_ground_pcd: Point Cloud consisting of non-ground points
            ground_indices: Indices of points that belong to ground from original Point Cloud
            non_ground_indices: Indices of points that belong to non-ground from original Point Cloud

        """

    # res = [0.01, 0.05, 0.09, 0.16, 0.25] 2
    # res = [0.01, 0.03, 0.05, 0.07, 0.09]  # 3 --best
    # res = [0.01, 0.05, 0.09, 0.16, 0.25, 0.38] #1
    # res = [0.01, 0.03, 0.05, 0.07, 0.09, 0.12] #4
    # res = [0.01, 0.03, 0.04]

    points = np.asarray(pcd.points)
    freq = np.full((len(points),), 0)

    # Applying HeightMap based ground segmentation for multiple resolutions
    for resolution in res:
        ground, non_ground, ground_indices, non_ground_indices = segment_ground_indices_ht(pcd,
                                                                                           resolution,
                                                                                           fwd_range,
                                                                                           side_range,
                                                                                           height_range,
                                                                                           height_threshold)
        freq[ground_indices] = freq[ground_indices] + 1

    freq = freq / len(res)
    ground_indices = np.where(freq >= 0.5)[0]
    non_ground_indices = np.where(freq < 0.5)[0]

    ground_pcd = pcd.select_by_index(ground_indices)
    non_ground_pcd = pcd.select_by_index(ground_indices, invert=True)

    return ground_pcd, non_ground_pcd, ground_indices, non_ground_indices
