import numpy as np


def get_batches(poses, batch_size):
    """
           Creates smaller batches for larger sequences

           Args:
               poses (List: nx4x4): Poses for the LiDAR scans
               batch_size (int): Size of a single batch in meters

           Returns:
               val: List of batches containing the starting and ending index for each batch
               len: Length of batch size + 100
    """
    length = batch_size

    curr_dist = 0
    start_idx = 0
    val = []

    prev_pose = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]

    for i in range(len(poses)):
        curr_pose = poses[i]
        dist = np.sqrt((curr_pose[0][3] - prev_pose[0][3]) ** 2 + (curr_pose[1][3] - prev_pose[1][3]) ** 2 + (
                curr_pose[2][3] - prev_pose[2][3]) ** 2)
        curr_dist = curr_dist + dist

        if curr_dist >= length:
            end_idx = i
            val.append([start_idx, end_idx])
            start_idx = i + 1
            curr_dist = 0

        prev_pose = curr_pose
    val.append([start_idx, len(poses)])
    return val, length + 100
