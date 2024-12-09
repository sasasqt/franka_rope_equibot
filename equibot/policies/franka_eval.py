# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# NOTE: Import here your extension examples to be propagated to ISAAC SIM Extensions startup
from equibot.policies.utils.misc import get_agent
# from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D <- this (in get_agent) caused BUG Windows fatal exception: access violation, if imported after simulationApp

# [Warning] [omni.isaac.kit.simulation_app] Modules: ['omni.kit_app'] were loaded before SimulationApp was started and might not be loaded correctly.
# [Warning] [omni.isaac.kit.simulation_app] Please check to make sure no extra omniverse or pxr modules are imported before the call to SimulationApp(...)
# not my fault?!
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import nest_asyncio
nest_asyncio.apply()

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.examples")
from franka_rope import IsaacUIUtils, VRUIUtils
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from .eval_util import *

# IsaacUIUtils.setUp()
# VRUIUtils.setUp()

import hydra
import os
from glob import glob

async def eval_async(ckpt_paths,agent,cfg):
    for ckpt_path in ckpt_paths:
        ckpt_name = ckpt_path.split("/")[-1].split(".")[0]
        agent.load_snapshot(ckpt_path)
        
        log_dir = os.getcwd()
        await EvalUtils.eval_async(
                agent,
                num_episodes=cfg.training.num_eval_episodes,
                log_dir=log_dir,
                reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
                ckpt_name=ckpt_name,
                cfg=cfg.franka_rope,
                simulation_app=simulation_app
            )
        
@hydra.main(config_path="configs", config_name="franka_base")
def main(cfg):
    # IsaacUIUtils.setUp()
    # from viztracer import VizTracer
    # tracer = VizTracer(tracer_entries=99999999,output_file="my_trace.json")
    # tracer.start()
    # TODO BUG Windows fatal exception: access violation
    agent = get_agent(cfg.agent.agent_name)(cfg)
    # tracer.stop()
    # tracer.save()
    agent.train(False)
    if os.path.isdir(cfg.training.ckpt):
        ckpt_dir = os.path.join(os.getcwd(),cfg.training.ckpt)
        ckpt_paths = list(glob(os.path.join(ckpt_dir, "ckpt*.pth")))
        assert len(ckpt_paths) >= cfg.eval.num_ckpts_to_eval
        ckpt_paths = list(sorted(ckpt_paths))[-cfg.eval.num_ckpts_to_eval :]
        assert f"{cfg.eval.last_ckpt}" in ckpt_paths[-1]
    else:
        ckpt_paths = [cfg.training.ckpt]
    
    asyncio.ensure_future(eval_async(ckpt_paths,agent,cfg))

    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()





# def eval(agent, num_episodes=1,log_dir=None,reduce_horizon_dim=True,ckpt_name=None):
#     print(">>> async eval <<<")

#     if hasattr(agent, "obs_horizon") and hasattr(agent, "ac_horizon"):
#         obs_horizon = agent.obs_horizon
#         ac_horizon = agent.ac_horizon
#         pred_horizon = agent.pred_horizon
#     else:
#         obs_horizon = 1
#         ac_horizon = 1
#         pred_horizon = 1

#     def _post_setup():
#         sample=EvalUtils._sample
#         world=sample._world
#         task=sample._task["Right"]
#         scene=world.scene
#         rope=sample._rope
#         robot=sample._robot["Right"]
#         robot_name = sample._robot_name["Right"]
#         target_name = sample._target_name["Right"]

#         def _post_reset():
#             obs_history = []
#             franka_end_effectors_pos=scene.get_object(robot_name).end_effector.get_world_pose()[0]
#             franka_end_effectors_rot=scene.get_object(robot_name).end_effector.get_world_pose()[1]
#             col1,col3=q2cols(franka_end_effectors_rot)
#             gravity_dir=[0,0,-1]
#             if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.5 is opened
#                 gripper_pose=0
#             else:
#                 gripper_pose=1
#             obs = dict(
#                 # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
#                 pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
#                 # state= eef_pos in saved npz
#                 state=np.array([[franka_end_effectors_pos[0],franka_end_effectors_pos[1],franka_end_effectors_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])

#             ) #pc and eef_pose
#             # print(">>> obs ",obs)
#             for i in range(obs_horizon):
#                 obs_history.append(obs)

#             while simulation_app.is_running():
#                 # make obs for agent
#                 if obs_horizon == 1 and reduce_horizon_dim:
#                     agent_obs = obs
#                 else:
#                     agent_obs = dict()
#                     for k in obs.keys():
#                         if k == "pc":
#                             # point clouds can have different number of points
#                             # so do not stack them
#                             agent_obs[k] = [o[k] for o in obs_history[-obs_horizon:]]
#                         else:
#                             agent_obs[k] = np.stack(
#                                 [o[k] for o in obs_history[-obs_horizon:]]
#                             )

#                 # predict actions
#                 st = time.time()
#                 # print(agent_obs)
#                 # print(agent_obs['pc'][0][0])
#                 # print(type(agent_obs['pc'][0][0]))
#                 # print(agent_obs['state'])
#                 # print(agent_obs['state'].shape,len(agent_obs['state'].shape))
                
#                 ac = agent.act(agent_obs, return_dict=False)
#                 logging.info(f"Inference time: {time.time() - st:.3f}s")

#                 # take actions
#                 for ac_ix in range(ac_horizon):
#                     if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
#                         break
#                     agent_ac = ac[ac_ix] if len(ac.shape) > 1 else ac

#                     update_action(agent_ac,scene.get_object(target_name),scene.get_object(robot_name).end_effector,robot._gripper)
#                     # Convenience function to step the application forward one frame
#                     print(">>>> before app update <<<<")
#                     simulation_app.update()
#                     franka_end_effectors_pos=scene.get_object(robot_name).end_effector.get_world_pose()[0]
#                     print("pos after update: ", franka_end_effectors_pos)
#                     franka_end_effectors_rot=scene.get_object(robot_name).end_effector.get_world_pose()[1]
#                     col1,col3=q2cols(franka_end_effectors_rot)
#                     gravity_dir=[0,0,-1]
#                     if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.5 is opened
#                         gripper_pose=0
#                     else:
#                         gripper_pose=1
#                     obs = dict(
#                         # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
#                         pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
#                         # state= eef_pos in saved npz
#                         state=np.array([[franka_end_effectors_pos[0],franka_end_effectors_pos[1],franka_end_effectors_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])

#                     )

#                     obs_history.append(obs)
#                     if len(obs) > obs_horizon:
#                         obs_history = obs_history[-obs_horizon:]
                        
#                     # task.set_params(pass)
#         sample._pre_physics_callback=_post_reset
    
#         EvalUtils.reset(callback_fn=_post_reset)

#     EvalUtils.setUp(callback_fn=_post_setup)
#     print(">>> post setup <<<")


if __name__ == "__main__":
    main()



















