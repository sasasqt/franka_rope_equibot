import omni.kit.app
from franka_rope import ControlFlow

import asyncio
import numpy as np
import time
from math import sqrt
import logging
from functools import partial

# singleton
class EvalUtils(ControlFlow):
    @classmethod
    async def _setup_async(cls,callback_fn=None):
        await omni.kit.app.get_app().next_update_async()
        await super().setUp_async(cls.cfg)
        if callback_fn is not None:
            callback_fn()

    @classmethod
    def _post_setup(cls):
        cls.sample=sample=cls._sample
        cls.world=world=sample._world
        cls.task=sample._task["Right"]
        cls.scene=world.scene
        cls.rope=sample._rope
        cls.robot=sample._robot["Right"]
        cls.robot_name = sample._robot_name["Right"]
        cls.target_name = sample._target_name["Right"]

    @classmethod
    async def _reset_async(cls,callback_fn=None):
        try:
            pass # TODO save data cls._sample._on_save_data_event()
        except:
            pass
        await super().reset_async()
        await cls._sample._on_follow_target_event_async(True)
        await cls._sample._world.play_async()
        await asyncio.sleep(5) # let the rope settle
        cls._sample._on_logging_event(True)
        cls.obs_history=[]
        cls.done=False
        cls.count=-1
        scene=cls.scene
        rope=cls.rope
        robot_name=cls.robot_name
        franka_end_effectors_pos=scene.get_object(robot_name).end_effector.get_world_pose()[0]
        franka_end_effectors_rot=scene.get_object(robot_name).end_effector.get_world_pose()[1]
        col1,col3=q2cols(franka_end_effectors_rot)
        gravity_dir=cls.gravity_dir=[0,0,-1]
        if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.05 is opened
            gripper_pose=0
        else:
            gripper_pose=1
        obs = dict(
            # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
            pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
            # state= eef_pos in saved npz
            state=np.array([[franka_end_effectors_pos[0],franka_end_effectors_pos[1],franka_end_effectors_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])

        ) #pc and eef_pose
        for i in range(cls.obs_horizon):
            cls.obs_history.append(obs)
        
        cls.sample._pre_physics_callback=partial(cls._post_reset,_onDone_async=cls._reset_async)
        # await cls._sample._world.pause_async()
        if callback_fn is not None:
            callback_fn()

    @classmethod
    def _post_reset(cls,step_size=None,_onDone_async=None):
        if cls.done:
            asyncio.ensure_future(_onDone_async(cls))
            return
        
        cls.count+=1
        sample=cls.sample
        world=cls.world
        task=cls.task
        scene=cls.scene
        rope=cls.rope
        robot=cls.robot
        robot_name=cls.robot_name
        target_name=cls.target_name
        obs_history = cls.obs_history
    
        agent=cls.agent
        obs_horizon=cls.obs_horizon
        ac_horizon=cls.ac_horizon
        pred_horizon=cls.pred_horizon
        reduce_horizon_dim=cls.reduce_horizon_dim
        gravity_dir=cls.gravity_dir
        done=cls.done

        # Make obs
        franka_end_effectors_pos=scene.get_object(robot_name).end_effector.get_world_pose()[0]
        print("pos after update: ", franka_end_effectors_pos)
        franka_end_effectors_rot=scene.get_object(robot_name).end_effector.get_world_pose()[1]
        col1,col3=q2cols(franka_end_effectors_rot)
        gravity_dir=[0,0,-1]
        if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.5 is opened
            gripper_pose=0
        else:
            gripper_pose=1
        obs = dict(
            # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
            pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
            # state= eef_pos in saved npz
            state=np.array([[franka_end_effectors_pos[0],franka_end_effectors_pos[1],franka_end_effectors_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])
        )

        obs_history.append(obs)
        if len(obs) > obs_horizon:
            obs_history = obs_history[-obs_horizon:]
            
        # make obs for agent
        if obs_horizon == 1 and reduce_horizon_dim:
            agent_obs = obs
        else:
            agent_obs = dict()
            for k in obs.keys():
                if k == "pc":
                    # point clouds can have different number of points
                    # so do not stack them
                    agent_obs[k] = [o[k] for o in obs_history[-obs_horizon:]]
                else:
                    agent_obs[k] = np.stack(
                        [o[k] for o in obs_history[-obs_horizon:]]
                    )

        # predict actions
        st = time.time()
        # print(agent_obs)
        # print(agent_obs['pc'][0][0])
        # print(type(agent_obs['pc'][0][0]))
        # print(agent_obs['state'])
        # print(agent_obs['state'].shape,len(agent_obs['state'].shape))
        if cls.count % ac_horizon == 0:
            cls.ac=ac = agent.act(agent_obs, return_dict=False)
        logging.info(f"Inference time: {time.time() - st:.3f}s")

        ac=cls.ac
        # take actions
        if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
            return
        agent_ac = ac[cls.count% ac_horizon] if len(ac.shape) > 1 else ac
        update_action(agent_ac,scene.get_object(target_name),scene.get_object(robot_name).end_effector,robot._gripper)

    @classmethod
    async def eval_async(cls,agent,num_episodes,log_dir,reduce_horizon_dim,ckpt_name,cfg,simulation_app):
        cls.agent=agent
        cls.num_episodes=num_episodes
        cls.log_dir=log_dir
        cls.reduce_horizon_dim=reduce_horizon_dim
        cls.ckpt_name=ckpt_name
        cls.cfg=cfg
        cls.simulation_app=simulation_app
        if hasattr(agent, "obs_horizon") and hasattr(agent, "ac_horizon"):
            cls.obs_horizon = agent.obs_horizon
            cls.ac_horizon = agent.ac_horizon
            cls.pred_horizon = agent.pred_horizon
        else:
            cls.obs_horizon = 1
            cls.ac_horizon = 1
            cls.pred_horizon = 1
        
        await cls._setup_async()
        cls._post_setup()
        await cls._reset_async()
        cls._post_reset(_onDone_async=cls._reset_async)

        
def update_action(agent_ac,target,eef,gripper):
    if agent_ac[0] <0.025:
        gripper.close()
    else:
        gripper.open()

    eef_world_pos=np.array(eef.get_world_pose()[0].tolist())
    eef_world_ori=np.array(eef.get_world_pose()[1].tolist())
    print("old pos: ", eef_world_pos)
    print("delta pos: ",agent_ac[1:1+3])
    tgt_pos=eef_world_pos+np.array(agent_ac[1:1+3])
    tgt_ori=quat_mul(normalize_quat(np.array(rpy2quat(agent_ac[4:4+3]))),normalize_quat(eef_world_ori))
    target.set_world_pose(position=tgt_pos,orientation=tgt_ori)
    print("applied pos: ",tgt_pos)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 =  q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return [w,x,y,z]

def normalize_quat(q):
    norm=sqrt(float(q[0])**2+float(q[1])**2+float(q[2])**2+float(q[3])**2)
    q[0],q[1],q[2],q[3]=q[0]/norm,q[1]/norm,q[2]/norm,q[3]/norm
    return q

def q2cols(q):
    q=normalize_quat(q)
    w,x,y,z=q
    col1=[2*(w**2+x**2)-1,2*(x*y+w*z),2*(x*z-w*y)]
    col3=[2*(w*y+x*z),2*(y*z-w*x),w**2-x**2-y**2+z**2]
    return col1,col3

import math

def rpy2quat(rpy):
    # return euler_angles_to_quat(rpy)
    roll, pitch, yaw=rpy[0],rpy[1],rpy[2]

    # Compute half angles
    half_roll = roll / 2.0
    half_pitch = pitch / 2.0
    half_yaw = yaw / 2.0

    # Compute trigonometric terms
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]