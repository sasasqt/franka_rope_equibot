import json
import open3d as o3d
import numpy as np
import os
from math import sqrt
import math
import hydra
import random
import h5py                               # pip install h5py
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

@hydra.main(config_path="equibot/policies/configs", config_name="franka_base")
def main(cfg):



    input_file = cfg.franka_rope.preprocess.input_dir
    output_dir = cfg.franka_rope.preprocess.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # rel = eval(str(cfg.franka_rope.preprocess.rel).title())
    # rpy = eval(str(cfg.franka_rope.preprocess.rpy).title())
    # flow = eval(str(cfg.franka_rope.preprocess.flow).title())

    gravity_dir = [0, 0, -1]  # z is up in isaac sim


    with h5py.File(input_file, "r") as f:          # open read-only
        pbar = tqdm(f['data'].keys(),
            total=len(f['data'].keys()),        # tqdm can now display % done
            desc="")
        for i,key in enumerate(f['data'].keys()):
            if i>99: return
            demo=f['data'][key]
            actions=demo['actions'] # (n,7) # dx dy dz, dr, dp, dy, -1/1
            obs=demo['obs']
            pc=obs['point_cloud'] # (n,\#pc,6) (pos+color)
            eef_pos=obs['robot0_eef_pos'] # (n,3)
            eef_xyzw=obs['robot0_eef_quat'] #(n,4)
        

            degrees = [random.randint(-5, 5),random.randint(-5, 5),random.randint(0, 360)]
            random_rotation = R.from_euler('xyz', degrees, degrees=True).as_matrix()
            random_translation = np.array([random.randint(-2, 4),random.randint(-2, 4),0])
            
            random_rotation=np.eye(3)
            random_translation=np.zeros(3)

            new_eef_pos=np.einsum('ij,nj->ni', random_rotation, eef_pos)+random_translation
            new_eef_ori=np.einsum('ij,njk->nik', random_rotation, R.from_quat(eef_xyzw,scalar_first=False).as_matrix())


            new_pc=pc[...,:3]
            new_pc=np.einsum('ij,ncj->nci', random_rotation, new_pc)+random_translation
            new_rgb=pc[...,3:]

            z_values = new_pc[:, :, 2]  # extract Z axis values

            percentiles = [1, 50, 99]  # 1st, median, 99th percentiles as example

            percentile_results = {}
            for p in percentiles:
                percentile_results[f'p{p}'] = np.percentile(z_values, p, axis=1)  # shape (b,)
                print(percentile_results)
                print(new_rgb.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_pc[55])
            pcd.colors = o3d.utility.Vector3dVector(new_rgb[55])
            o3d.visualization.draw_geometries([pcd])

            new_pc=np.concatenate((new_pc,new_rgb),axis=-1)
            # p1 = np.percentile(z_values, 5, axis=1)
            # p99 = np.percentile(z_values, 95, axis=1)
            # counts_below_p1 = np.sum(z_values <= p1[:, None], axis=1)  # count points below or equal 1st percentile per batch
            # counts_above_p99 = np.sum(z_values >= p99[:, None], axis=1)  # count points above or equal 99th percentile per batch
            # print(counts_below_p1,counts_above_p99)
            new_delta_pos = actions[...,:3]
            new_delta_ori=R.from_euler('xyz', actions[...,3:6], degrees=False).as_matrix()
            new_gripper_action=actions[...,6]
            
            for _i in range(new_pc.shape[0]):

                ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
                cols = [new_eef_ori[_i,i, j] for i, j in ori_indices]
                eef_pos = np.array((
                        new_eef_pos[_i][0],
                        new_eef_pos[_i][1],
                        new_eef_pos[_i][2],
                        cols[0],
                        cols[1],
                        cols[2],
                        cols[3],
                        cols[4],
                        cols[5],
                        gravity_dir[0],
                        gravity_dir[1],
                        gravity_dir[2],
                        -1 if _i==0 else new_gripper_action[_i-1],
                    ))
                mat4x4 = np.eye(4)
                mat4x4[:3, :3] = new_delta_ori[_i]
                mat4x4[:3, 3] = new_delta_pos[_i]
                mat4x4[3,3]=new_gripper_action[_i]

                action = np.array(
                    mat4x4
                )
                # np.savez(
                #     # :02d is expected from the dataset py
                #     os.path.join(output_dir + rf"/01_ep{i:06d}_view0_t{_i:02d}.npz"),
                #     pc=new_pc[_i], # (40, 6) = (num_points, src + tgt)
                #     eef_pos=eef_pos, #  (13,)
                #     action=np.array(action[np.newaxis, :]), #  (1, 4, 4)

                # )
            pbar.update(1)



if __name__ == "__main__":
    main()
