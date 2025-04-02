import h5py

f = h5py.File('/home/workstation/project/equidiff/data/robomimic/datasets/coffee_d0/coffee_d0.hdf5', "r")

# data -> demo_x -> actions', 'dones', 'obs', 'rewards', 'states'

# obs:
# <KeysViewHDF5 ['agentview_depth', 'agentview_image', 'birdview_depth', 'birdview_image', 'object', 'point_cloud', 'robot0_eef_pos', 'robot0_eef_pos_rel_pod', 'robot0_eef_pos_rel_pod_holder', 'robot0_eef_quat', 'robot0_eef_quat_rel_pod', 'robot0_eef_quat_rel_pod_holder', 'robot0_eef_vel_ang', 'robot0_eef_vel_lin', 'robot0_eye_in_hand_depth', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'sideview_depth', 'sideview_image', 'voxels']>

# obs
#     point_cloud shape (227, 1024, 6),
#     voxels shape (227, 4, 64, 64, 64)
#     robot0_eef_pos shape (227, 3)
#     robot0_eef_quat shape (227, 4) # appears to be (x, y, z, w)
#     robot0_gripper_qpos/qvel shape (219, 2)
# actions shape (227, 7)


#  print(f['data']['demo_0']['obs']['point_cloud'])






