import numpy as np
import time
from math import sqrt
import logging
from functools import partial

import os
import pathlib

# from omni.kit.capture.viewport import CaptureOptions, CaptureExtension, CaptureStatus
# import omni.renderer_capture
from itertools import product
from datetime import datetime

from equibot.policies.etseed_train import init_model_and_optimizer, prepare_model_input, prepare_model_output

from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler

from scipy.spatial.transform import Rotation as R # this operates on float64, unlike kornia which is on float32
import torch
import wandb
# TODO dont block the ui: put the inference code in a new process, and cross processes communication
# TODO the objective metrics?
from collections import defaultdict
import socket
import json
import time

# singleton
class EvalUtils():

    @classmethod
    def eval(cls,config,obs_history=None):
        import torch

        def convert_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor(item) for item in obj]
            else:
                return obj

        nets=config['nets']
        noise_scheduler=config['noise_scheduler']
        gripper_noise_scheduler=config['gripper_noise_scheduler']
        if config['arch']==0:
            from equibot.policies.etseed_test import test_batch
        # elif config['arch']==1:
        #     pass
        elif config['arch']==2:
            from equibot.policies.etseed_test_sep2 import test_batch
        elif config['arch']==3:
            from equibot.policies.etseed_test_sep_no_diffusion_no_gripper import test_batch
        elif config['arch']==4:
            from equibot.policies.etseed_test_sep2_nounetdiffusion import test_batch
        else:
            raise NotImplementedError
        

        obs_horizon=config['obs_horizon']
        ac_horizon=config['action_horizon']
        pred_horizion=config['pred_horizon']
        gravity_dir = [0, 0, -1]

        while True:
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.connect(('localhost', 9999))
                print("Client (B) connected to server.")
                break  # Exit loop on success
            except ConnectionRefusedError:
                print("Connection refused, retrying in 2 seconds...")
                time.sleep(2)  # Wait before retrying

        with client:

            while True:
                time.sleep(0.1)

                # B receives first
                data = b''
                while True:
                    packet = client.recv(10240000)
                    print("receiving")
                    if not packet:
                        break
                    data += packet
                    if packet.endswith(b"\n"):
                        data = data[:-1]
                        break

                if data:
                    obs_history = json.loads(data.decode('utf-8'))
                    print("Received: obs_history", len(obs_history))
                else:
                    print("No obs_history received.")
                    continue
                
                new_pcs=[]
                new_eef_poses=[]
                for obs in obs_history:
                    
                    # Inputs coming from robomimic env
                    pc=obs['point_cloud'] # (b,obs_horizon,\#pc,6) (pos+color)
                    eef_pos=obs['robot0_eef_pos'] # (b,obs_horizon,3)
                    eef_xyzw=obs['robot0_eef_quat'] #(b,obs_horizon,4)
                    past_gripper_action=obs['past_gripper_action'] #(b,obs_horizon,1)
                    # test_batch expects following inputs:
                    # pc: [B,obs_horizon,num_pts,6]
                    # eef_pos(is pose not position): [B, obs_horizon, num_eef, pose gripper action etc]
                    # it outputs: action: [B,pred_horizon,4by4]
                    
                    new_pc=np.array(pc)
                    B=new_pc.shape[0]
                    N=new_pc.shape[1]
                    new_eef_pos=np.array(eef_pos)
                    # print(new_pc.shape,'new_pc')
                    # print(new_eef_pos.shape,'new_eef_pos')
                    # print(np.array(eef_xyzw).shape,'eef_xyzw')
                    # print(np.array(past_gripper_action).shape,'past_gripper_action')
                    new_eef_ori=R.from_quat(np.array(eef_xyzw).reshape(-1,4),scalar_first=False).as_matrix()
                    new_eef_ori=new_eef_ori.reshape(B,-1,3,3)
                    # print(np.array(new_eef_ori).shape,'new_eef_ori')

                    new_eef_pose = np.zeros((B, N, 13), dtype=np.float32)  # or 13 if you add gripper

                    for _b in range(B):
                        for _n in range(N):

                            ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
                            cols = [new_eef_ori[_b,_n,i, j] for i, j in ori_indices]
                            eef_pose = np.array((
                                    new_eef_pos[_b,_n][0],
                                    new_eef_pos[_b,_n][1],
                                    new_eef_pos[_b,_n][2],
                                    cols[0],
                                    cols[1],
                                    cols[2],
                                    cols[3],
                                    cols[4],
                                    cols[5],
                                    gravity_dir[0],
                                    gravity_dir[1],
                                    gravity_dir[2],
                                    past_gripper_action[_b][_n][0],
                                ))
                            new_eef_pose[_b, _n] = eef_pose
                    new_pcs.append(new_pc)
                    new_eef_poses.append(new_eef_pose)
                stacked_pcs = np.concatenate(new_pcs, axis=1)  # shape [B, K*N, ...]
                stacked_eef_poses = np.concatenate(new_eef_poses, axis=1)  # shape [B, K*N, ...]

                agent_obs=dict()
                agent_obs['pc']=torch.from_numpy(stacked_pcs)
                agent_obs['eef_pos']=torch.from_numpy(stacked_eef_poses)
                # print(agent_obs['eef_pos'].shape,"agent_obs['eef_pos']agent_obs['eef_pos']")
                st = time.time()

                ac = test_batch(nets=nets, noise_scheduler=noise_scheduler,gripper_noise_scheduler=gripper_noise_scheduler, nbatch=agent_obs, device='cuda:0',config=config,isVisualEval=True)
                # print(ac.shape, "ac?") # b Ha 4 4

                delta_xyz = ac[:, :, :3, 3]
                rotations=ac[:, :,:3, :3].cpu()
                gripper_action=ac[:,:,3,3].unsqueeze(-1)

                M=delta_xyz.shape[1]
                # print(delta_xyz.shape,rotations.shape,gripper_action.shape,torch.zeros(B,M,3).shape)
                # delta_xyzw = R.from_matrix(rotations.reshape(-1,3,3)).as_quat(scalar_first=False,canonical=False) #kornia.geometry.conversions.rotation_matrix_to_quaternion(rotations)
                # delta_xyzw=torch.from_numpy(delta_xyzw.reshape(B,-1,4)).cuda()
                delta_euler = R.from_matrix(rotations.reshape(-1,3,3)).as_euler('xyz') #kornia.geometry.conversions.rotation_matrix_to_quaternion(rotations)
                delta_euler=torch.from_numpy(delta_euler.reshape(B,-1,3)).cuda()
                rel_action=torch.cat([delta_xyz*1.0,delta_euler,gripper_action],dim=-1)

                print(delta_xyz)
                print("--")
                print(gripper_action)
                print("!!!")
                # x towards screen
                # y right
                # z up
                # ccw around x
                # ccw around y
                # ccw around z
                # -1 open 1 close

                print("B replies: action dict")
                json_str = json.dumps(convert_tensor(rel_action))+ '\n'
                encoded = json_str.encode('utf-8')

                client.sendall(encoded+b'\n')
                # client.shutdown(socket.SHUT_WR)

                # expected # (B, n_action_steps, act_dim)
                logging.info(f"Inference time: {time.time() - st:.3f}s")

        
    
    