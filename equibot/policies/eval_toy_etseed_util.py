import omni.kit.app
from franka_pick import ControlFlow

from omni.isaac.core.utils.types import ArticulationAction

import asyncio
import numpy as np
import time
from math import sqrt
import logging
from functools import partial

from omni.isaac.utils._isaac_utils import math as mu
import os
import pathlib
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
from video_encoding import get_video_encoding_interface

# from omni.kit.capture.viewport import CaptureOptions, CaptureExtension, CaptureStatus
# import omni.renderer_capture
from itertools import product
from datetime import datetime

from pxr import Gf, UsdGeom
import omni.usd

from equibot.policies.etseed_train import init_model_and_optimizer, prepare_model_input, prepare_model_output

from equibot.policies.utils.etseed.utils.SE3diffusion_scheduler import DiffusionScheduler

from scipy.spatial.transform import Rotation as R # this operates on float64, unlike kornia which is on float32
import torch
import wandb
# TODO dont block the ui: put the inference code in a new process, and cross processes communication
# TODO the objective metrics?
from collections import defaultdict
import json

# singleton
class EvalUtils(ControlFlow):
    _end=600

    @classmethod
    async def _setup_async(cls,callback_fn=None):
        await omni.kit.app.get_app().next_update_async()
        await super().setUp_async(cls.cfg)
        if callback_fn is not None:
            callback_fn()

    @classmethod
    def _post_setup(cls):
        cls._end=int(cls.cfg.max_end) or cls._end
        cls.sample=sample=cls._sample
        cls.world=world=sample._world
        cls.task=sample._task["Left"]
        cls.scene=world.scene
        # cls.rope=sample._rope
        cls.robot=sample._robot["Left"]
        cls.robot_name = sample._robot_name["Left"]
        cls.target_name = sample._target_name["Left"]
        cls.obs_history=[]
        cls.done=False
        cls.count=-1

    @classmethod
    async def _reset_async(cls,callback_fn=None):

        try:
            pass # TODO save data cls._sample._on_save_data_event()
        except:
            pass
        await super().reset_async()
        await cls._sample._world.play_async()

        sample=cls.sample
        world=cls.world
        task=cls.task
        scene=cls.scene
        # rope=cls.rope
        robot=cls.robot
        robot_name=cls.robot_name
        target_name=cls.target_name
        cls.gravity_dir=[0,0,-1]

        # # cube at the end pick v1
        # cls._tgt_pc = np.array(
        #             [
        #                 [
        #                     0.02616778388619423,
        #                     -0.17271608114242554,
        #                     0.060000933706760406
        #                 ],
        #                 [
        #                     0.026168517768383026,
        #                     -0.17271505296230316,
        #                     9.35697755721776e-07
        #                 ],
        #                 [
        #                     0.017845619469881058,
        #                     -0.23213613033294678,
        #                     0.05999980866909027
        #                 ],
        #                 [
        #                     -0.033252257853746414,
        #                     -0.16439391672611237,
        #                     0.060000352561473846
        #                 ],
        #                 [
        #                     0.017846353352069855,
        #                     -0.2321351021528244,
        #                     -1.8844585270016978e-07
        #                 ],
        #                 [
        #                     -0.03325152397155762,
        #                     -0.16439288854599,
        #                     3.523719840359263e-07
        #                 ],
        #                 [
        #                     -0.041574422270059586,
        #                     -0.2238139659166336,
        #                     0.05999922752380371
        #                 ],
        #                 [
        #                     -0.04157368838787079,
        #                     -0.22381293773651123,
        #                     -7.717716243860195e-07
        #                 ]
        #             ]
        #     )

        # cube at the beginning: toymix
        cls._tgt_pc = np.array(
                 [
                        [
                            0.030000003054738045,
                            -0.12000000476837158,
                            0.06000000238418579
                        ],
                        [
                            0.030000003054738045,
                            -0.12000000476837158,
                            3.725290298461914e-09
                        ],
                        [
                            0.030000003054738045,
                            -0.18000000715255737,
                            0.06000000238418579
                        ],
                        [
                            -0.029999995604157448,
                            -0.12000000476837158,
                            0.06000000238418579
                        ],
                        [
                            0.030000003054738045,
                            -0.18000000715255737,
                            3.725290298461914e-09
                        ],
                        [
                            -0.029999995604157448,
                            -0.12000000476837158,
                            3.725290298461914e-09
                        ],
                        [
                            -0.029999995604157448,
                            -0.18000000715255737,
                            0.06000000238418579
                        ],
                        [
                            -0.029999995604157448,
                            -0.18000000715255737,
                            3.725290298461914e-09
                        ]
                    ]
            )
        # random_rotation=np.eye(3)
        random_translation=np.zeros(3)
        # rotate world first after play(), otherwise the franka will compensate the rotation somehow in their code
        # rotate in simulation not in usd
        if cls.cfg.rotation is not None:
            sample._world_xform.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(list(cls.cfg.rotation)))
            _rotation = R.from_euler('xyz', cls.cfg.rotation, degrees=True).as_matrix()
            cls._tgt_pc=np.array([_rotation@vector for vector in cls._tgt_pc])

        if cls.cfg.translation is not None:
            sample._world_xform.GetAttribute('xformOp:translate').Set(Gf.Vec3f(list(cls.cfg.translation))) # GetAttribute is only callable for usd objects defined via stage.DefinePrim, not for UsdGeom.Xform
            _translation = cls.cfg.translation
            cls._tgt_pc=np.array([vector+_translation for vector in cls._tgt_pc])

        if cls.cfg.scale is not None:
            sample._world_xform.GetAttribute('xformOp:scale').Set(Gf.Vec3f(list(cls.cfg.scale)))
            _scale=cls.cfg.scale
            cls._tgt_pc=np.array([vector*_scale for vector in cls._tgt_pc])

            await omni.kit.app.get_app().next_update_async()
            # await asyncio.sleep(3) # BUG weird concurrent issue, otherwise shape undo for the scene rotation (in simulation)
            await omni.kit.app.get_app().next_update_async()

        # cls._tgt_pc=np.array([vector+np.array([0.1,0,0]) for vector in cls._tgt_pc]) # ood
   
        import omni
        import omni.usd
        from pxr import Sdf
        import omni.kit.commands

        omni.kit.commands.execute('ChangeProperty',
            prop_path=Sdf.Path('/World/defaultGroundPlane/Environment/Geometry.xformOp:orient'),
            value=Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)),
            prev=Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)),
            usd_context_name=omni.usd.get_context().get_stage())

        await asyncio.sleep(3)
        await cls._sample._on_follow_target_event_async(True)


        if cls.cfg.from_demo is None:
            extra_repeat=cls.obs_horizon-1
            # await cls._sample._world.play_async()
            await asyncio.sleep(3) # let the rope settle
            # await cls._sample._world.pause_async()

        else:
            extra_repeat=0
            data_logger=cls.data_logger=cls.sample._data_logger
            data_logger.load(log_path=cls.cfg.from_demo)
            demo_length=data_logger.get_num_of_data_frames()
            start_time=cls.start_time=cls.cfg.start_time or int(0.25*demo_length)

            if start_time>= demo_length:
                print(f">>> start time {start_time} excessed the demo length {demo_length}, reset to 1/4 of demo length <<<")
                start_time=int(0.25*demo_length)

            if start_time is not None:
                if start_time+1-cls.obs_horizon <0:
                    extra_repeat=cls.obs_horizon-start_time-1
                    # obs hon = 4: 
                    #   start =4 extra_repeat=0
                    #   start=3 extra_repeat=0
                    #   start=2 extra_repeat =1
                    #   start=1 extra_repeat=2
                    #   start=0 extra_repeat=3
                else: 
                    extra_repeat=0
            else:
                pass # TODO start time = the first frame where target moves

        cls.obs_history=[]
        cls.done=False
        cls.count=max(-20,-cls.start_time) if cls.cfg.from_demo is not None else -1 # see issue 1 and issue 2
        # await cls._sample._on_follow_target_event_async(True)
        # await asyncio.sleep(5)
        # await cls._sample._world.pause_async()

        for i in range(cls.obs_horizon-extra_repeat):
            # obs hon = 4: 
            #   start =4 data_frame_index=1+i extra_repeat=0 i=0-3
            #   start=3 data_frame_index=0+i extra_repeat=0 i=0-3
            #   start=2 data_frame_index =0+i extra_repeat=1 i=0-2
            #   start=1 data_frame_index=0+i extra_repeat=2 i=0-1
            #   start=0 data_frame_index=0+i extra_repeat=3 i=0-0
            if cls.cfg.from_demo is not None:
                data_frame = data_logger.get_data_frame(data_frame_index=start_time-cls.obs_horizon+1+extra_repeat+i)
                for idx,_str in enumerate(["Left"]):  
                    # world.scene.get_object(robot_name).set_joint_positions(
                    #     np.array(data_frame.data[_str][f"{_str}_joint_positions"])
                    # )
                    # await omni.kit.app.get_app().next_update_async()        

                    world.scene.get_object(target_name).set_local_pose(
                        translation=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                        orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                    )


                quat=np.array([data_frame.data["Cube"]["cube_world_orientation"]])
                cls._sample._cube.set_local_poses(
                    translations=np.array([data_frame.data["Cube"]["cube_world_position"]]),
                    orientations=np.array(quat),
                )
                xform=world.scene.get_object("/Sphere")
                xform.set_world_poses(positions=sample._cube.get_world_poses()[0],orientations=sample._cube.get_world_poses()[1])

                # rope.set_world_pose(
                #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
                # )


            Left_target_world_pos=scene.get_object(target_name).get_world_pose()[0]
            Left_target_world_rot=scene.get_object(target_name).get_world_pose()[1]
            Left_target_world_pos=scene.get_object(target_name).get_world_pose()[0]
            print("pos !!!!!!!!!!!: ", Left_target_world_pos)
            ori=R.from_quat(Left_target_world_rot,scalar_first=True).as_matrix()
            ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
            cols = [ori[i, j] for i, j in ori_indices]
            
            gravity_dir=cls.gravity_dir
            if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.05 is opened
                gripper_pose=0
            else:
                gripper_pose=1

            pc=[]
            i=0
            xform=world.scene.get_object("/Sphere")
            xform.set_world_poses(positions=sample._cube.get_world_poses()[0],orientations=sample._cube.get_world_poses()[1])

            while scene.object_exists(f'/Sphere/sphere{i}'):
                sphere=scene.get_object(f'/Sphere/sphere{i}')
                pc.append(sphere.get_world_pose()[0].tolist())
                i+=1
            pc=np.array(pc)
            tgt_pc=cls._tgt_pc

            pc=np.concatenate((pc,tgt_pc),axis=1)
            # if eval(str(cls.cfg.flow).title()):
            #     pc=np.concatenate((pc,tgt_pc-pc),axis=1) # [ 1.57756746e-01  9.57879238e-03  5.00003956e-02 -7.45579600e-04 -6.01215288e-04 -4.09781933e-07]
            # else:
            #     pc=np.concatenate((pc,tgt_pc),axis=0) 

            eef_pos = np.array((
                    Left_target_world_pos[0],
                    Left_target_world_pos[1],
                    Left_target_world_pos[2],
                    cols[0],
                    cols[1],
                    cols[2],
                    cols[3],
                    cols[4],
                    cols[5],
                    gravity_dir[0],
                    gravity_dir[1],
                    gravity_dir[2],
                    gripper_pose,
                ))

            obs = dict(
                # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
                pc=pc,
                # pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
                eef_pos=eef_pos,
                # state= eef_pos in saved npz
                # state=np.array([[Left_target_world_pos[0],Left_target_world_pos[1],Left_target_world_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])
            ) #pc and eef_pose

            if cls.obs_horizon-extra_repeat-1-i<=0:
                pass
                # await cls._sample._world.pause_async() 
            cls.obs_history.append(obs)
            if i==0:
                for _ in range(extra_repeat):
                    cls.obs_history.append(obs)


        if cls.cfg.from_demo is not None:
            data_frame = data_logger.get_data_frame(data_frame_index=start_time+1)
            for idx,_str in enumerate(["Left"]):   
                world.scene.get_object(robot_name).set_joint_positions(
                    np.array(data_frame.data[_str][f"{_str}_joint_positions"])
                )

            for idx,_str in enumerate(["Left"]):   
                world.scene.get_object(target_name).set_local_pose(
                    translation=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                    orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                )

            quat=np.array([data_frame.data["Cube"]["cube_world_orientation"]])
            cls._sample._cube.set_local_poses(
                translations=np.array([data_frame.data["Cube"]["cube_world_position"]]),
                orientations=np.array(quat),
            )

            xform=world.scene.get_object("/Sphere")
            xform.set_world_poses(positions=sample._cube.get_world_poses()[0],orientations=sample._cube.get_world_poses()[1])

                # rope.set_world_pose(
                #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
                # )

        # data_frame = data_logger.get_data_frame(data_frame_index=start_time-cls.obs_horizon+1+extra_repeat+i)
        # for idx,_str in enumerate(["Left"]):  
        #     world.scene.get_object(robot_name).set_joint_positions(
        #         np.array(data_frame.data[_str][f"{_str}_joint_positions"])
        #     )

        # nets, optimizer, lr_scheduler = init_model_and_optimizer(cls.device,cls.config)
        # noise_scheduler = DiffusionScheduler(num_steps=cls.config["diffusion_steps"],mode=cls.config["diffusion_mode"],device=cls.device)

        nets=cls.config['nets']
        noise_scheduler=cls.config['noise_scheduler']
        gripper_noise_scheduler=cls.config['gripper_noise_scheduler']


        cls._sample._on_logging_event(True)
        # await cls._sample._world.play_async()

        # await cls._sample._world.pause_async()
        if callback_fn is not None:
            callback_fn()

        stage = omni.usd.get_context().get_stage()

        alternative_camera_xform=UsdGeom.Xform.Define(stage, f'/World/Camera')
        alternative_camera_xform.AddTranslateOp().Set(Gf.Vec3f([0,0,0]))
        alternative_camera_xform.AddRotateXYZOp().Set(Gf.Vec3f([0,0,0]))

        alternative_camera_prim=UsdGeom.Camera.Define(stage, f'/World/Camera/Camera')
        alternative_camera_prim.AddTranslateOp().Set(Gf.Vec3f([5.0,-0.3,3.2]))
        alternative_camera_prim.AddRotateXYZOp().Set(Gf.Vec3f([57,0,85]))
        # UsdGeom.Camera.Define(stage, alternative_camera_path_str)
        get_active_viewport().camera_path='/World/Camera/Camera'
        
        cls._current_time=current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.log({"output_file": current_time})
        capture_filename = "captured"
        cls._output_folder = pathlib.Path(os.getcwd()).joinpath(capture_filename)
        cls.make_sure_directory_existed(cls._output_folder)
        cls.clean_files_in_directory(cls._output_folder, ".png")

        await omni.kit.app.get_app().next_update_async()
        # await asyncio.sleep(3) # BUG weird concurrent issue, otherwise shape undo for the scene rotation (in simulation)
        await omni.kit.app.get_app().next_update_async()

        cls.sample._pre_physics_callback=None


        cls.sample._pre_physics_callback=partial(cls._reset,nets,noise_scheduler,gripper_noise_scheduler,_onDone_async=cls._reset_async)
        
        await omni.kit.app.get_app().next_update_async()
        # await asyncio.sleep(3) # BUG weird concurrent issue, otherwise shape undo for the scene rotation (in simulation)
        await omni.kit.app.get_app().next_update_async()

        while cls.count<-1:
            await asyncio.sleep(0.01)


        cls.sample._pre_physics_callback=partial(cls._post_reset,nets,noise_scheduler,gripper_noise_scheduler,_onDone_async=cls._reset_async)

        # # see extscache\omni.kit.capture.viewport-1.5.1\omni\kit\capture\viewport\tests\test_capture_png.py
        # cls._capture_instance = CaptureExtension().get_instance()

        # capture_filename = "captured"
        # filePath = pathlib.Path(os.getcwd()).joinpath(capture_filename)
        # options = CaptureOptions()
        # options.file_type = ".png"
        # options.start_frame = 0
        # options.end_frame = cls._end+5 # it pauses the simulation after the end of recording
        # options.capture_every_Nth_frames = 1
        # options.output_folder = str(filePath)
        # print(f">>> video path: {str(filePath)}")
        # options.file_name = f"{current_time}"
        # options.overwrite_existing_frames = True
        # cls.make_sure_directory_existed(options.output_folder)
        # options.hdr_output = False
        # viewport_api=get_active_viewport()
        # options.camera = viewport_api.camera_path.pathString
        # cls._output_folder=options.output_folder
        # cls._capture_instance.options = options


    # this captures the entire omniverse window
    # # see extscache\omni.kit.renderer.capture-0.0.0+10a4b5c0.wx64.r.cp310\omni\kit\renderer_capture\_renderer_capture.pyi
    # def capture(path=os.getcwd(),image_name="output.png"):
    #     image1 = pathlib.Path(path,image_name)
    #     print(str(image1))
    #     omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(image1))
    #     omni.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()

    def make_sure_directory_existed(directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as error:
                print(f"Directory cannot be created: {dir}")
                return False
        return True


    def clean_files_in_directory(directory, suffix=".png"):
        if not os.path.exists(directory):
            return
        images = os.listdir(directory)
        for item in images:
            if item.endswith(suffix):
                os.remove(os.path.join(directory, item))

    def viewport_capture(image_name: str, output_img_dir: str, viewport=None, use_log: bool = True):

        image1 = str(pathlib.Path(output_img_dir).joinpath(image_name))
        if use_log:
            print(f"Capturing {image1}")

        if viewport is None:
            viewport = get_active_viewport()

        return capture_viewport_to_file(viewport, file_path=image1)




    @classmethod
    def _reset(cls,nets, noise_scheduler,gripper_noise_scheduler,step_size=None,_onDone_async=None):
        if cls.config['arch']==0:
            from equibot.policies.etseed_test import test_batch
        # elif config['arch']==1:
        #     pass
        elif cls.config['arch']==2:
            from equibot.policies.etseed_test_sep2 import test_batch
        elif cls.config['arch']==3:
            from equibot.policies.etseed_test_sep_no_diffusion_no_gripper import test_batch
        elif cls.config['arch']==4:
            from equibot.policies.etseed_test_sep2_nounetdiffusion import test_batch
        else:
            raise NotImplementedError
        
        print("---")
        if step_size is None:
            pass # return
        if cls.done:
            asyncio.ensure_future(_onDone_async(cls))
            return
        if cls.count >= -1:
            return
        
        cls.count+=1

        # if cls.count==0:
        #     cls._capture_instance.start()
        # # cls.capture(image_name=f"{cls.count}.png")
        # if cls.count>=0 and cls.count<cls._end:
        #     cls.viewport_capture(image_name=f"{cls._current_time}_{cls.count}.png", output_img_dir=cls._output_folder)

        sample=cls.sample
        world=cls.world
        task=cls.task
        scene=cls.scene
        # rope=cls.rope
        robot=cls.robot
        robot_name=cls.robot_name
        target_name=cls.target_name
        obs_history = cls.obs_history


        obs_horizon=cls.obs_horizon
        ac_horizon=cls.ac_horizon
        pred_horizion=cls.pred_horizon
        reduce_horizon_dim=cls.reduce_horizon_dim

        print(cls.count)
        # ISSUE 1: need at least 2 dt to populate simulation physics when gripper is holding the rope
        # ISSUE 2: during the initial alignment, the hand/gripper need to rotate to match the orientation of the taget cube
        if cls.cfg.from_demo is not None and cls.count < 0 and cls.count>=-2:
            data_frame = cls.data_logger.get_data_frame(data_frame_index=cls.start_time+1+cls.count)
            for idx,_str in enumerate(["Left"]):  
                world.scene.get_object(target_name).set_local_pose(
                    translation=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                    orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                )

                quat=np.array([data_frame.data["Cube"]["cube_world_orientation"]])
                cls._sample._cube.set_local_poses(
                    translations=np.array([data_frame.data["Cube"]["cube_world_position"]]),
                    orientations=np.array(quat),
                )
                xform=world.scene.get_object("/Sphere")
                xform.set_world_poses(positions=sample._cube.get_world_poses()[0],orientations=sample._cube.get_world_poses()[1])

                # rope.set_world_pose(
                #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
                # )
            return

    @classmethod
    def _post_reset(cls,nets, noise_scheduler,gripper_noise_scheduler,step_size=None,_onDone_async=None):
        if cls.config['arch']==0:
            from equibot.policies.etseed_test import test_batch
        # elif config['arch']==1:
        #     pass
        elif cls.config['arch']==2:
            from equibot.policies.etseed_test_sep2 import test_batch
        elif cls.config['arch']==3:
            from equibot.policies.etseed_test_sep_no_diffusion_no_gripper import test_batch
        elif cls.config['arch']==4:
            from equibot.policies.etseed_test_sep2_nounetdiffusion import test_batch
        else:
            raise NotImplementedError
        
        print("---")
        if step_size is None:
            pass # return
        if cls.done:
            asyncio.ensure_future(_onDone_async(cls))
            return
        
        cls.count+=1
        # if cls.count==0:
        #     cls._capture_instance.start()
        # # cls.capture(image_name=f"{cls.count}.png")
        if cls.count>=0 and cls.count<cls._end:
            cls.viewport_capture(image_name=f"{cls._current_time}_{cls.count}.png", output_img_dir=cls._output_folder)

        sample=cls.sample
        world=cls.world
        task=cls.task
        scene=cls.scene
        # rope=cls.rope
        robot=cls.robot
        robot_name=cls.robot_name
        target_name=cls.target_name
        obs_history = cls.obs_history


        obs_horizon=cls.obs_horizon
        ac_horizon=cls.ac_horizon
        pred_horizion=cls.pred_horizon
        reduce_horizon_dim=cls.reduce_horizon_dim

        if cls.count>cls._end:
            log_path=os.path.join(cls._output_folder, f"{cls._current_time}.json")
            print(log_path)
            cls._sample._on_save_data_event(log_path=log_path)
            with open(os.path.join(cls._output_folder, f'my{cls._current_time}.json'), 'w') as f:
                json.dump(myjson, f,  separators=(',', ':'))
            _frame_filenames=[]
            for i in range(cls.count-1):
                frame_path=os.path.join(cls._output_folder, f"{cls._current_time}_{i}.png")
                _frame_filenames.append(frame_path)

            g_video_encoding_api=get_video_encoding_interface()
            video_path_str=os.path.join(cls._output_folder, f"{cls._current_time}.mp4")
            g_video_encoding_api.start_encoding(video_path_str, round(1/float(eval(cls.cfg.rendering_dt))), len(_frame_filenames), True)
            for frame_filename in _frame_filenames:
                g_video_encoding_api.encode_next_frame_from_file(frame_filename)
            g_video_encoding_api.finalize_encoding()

            cls.clean_files_in_directory(cls._output_folder, ".png")

            wandb.finish()
            cls.simulation_app.close()

        print(cls.count)
        # # ISSUE 1: need at least 2 dt to populate simulation physics when gripper is holding the rope
        # # ISSUE 2: during the initial alignment, the hand/gripper need to rotate to match the orientation of the taget cube
        # if cls.cfg.from_demo is not None and cls.count < 0 and cls.count>=-2:
        #     data_frame = cls.data_logger.get_data_frame(data_frame_index=cls.start_time+1+cls.count)
        #     for idx,_str in enumerate(["Left"]):  
        #         world.scene.get_object(target_name).set_world_pose(
        #             position=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
        #             orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
        #         )

        #         world.scene.get_object('vbar').set_world_pose(
        #             position=np.array(data_frame.data["T"]["vbar_world_position"]),
        #             orientation=np.array(data_frame.data["T"]["vbar_world_orientation"]),
        #         )
        #         world.scene.get_object('hbar').set_world_pose(
        #             position=np.array(data_frame.data["T"]["hbar_world_position"]),
        #             orientation=np.array(data_frame.data["T"]["hbar_world_orientation"]),
        #         )

        #         # rope.set_world_pose(
        #         #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
        #         #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
        #         # )
        #     return

        if cls.count < 0:
            return
        
        # Make obs
        Left_target_world_pos=scene.get_object(target_name).get_world_pose()[0]
        print("pos obs-ed: ", Left_target_world_pos)
        Left_target_world_rot=scene.get_object(target_name).get_world_pose()[1]
        ori=R.from_quat(Left_target_world_rot,scalar_first=True).as_matrix()
        ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
        cols = [ori[i, j] for i, j in ori_indices]
        gravity_dir=[0,0,-1]
        if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.05 is opened
            gripper_pose=0
        else:
            gripper_pose=1

        gripper_state="CLOSED" if gripper_pose==0 else "opened"
        print(f"gripper is {gripper_state}")
    
        pc=[]
        i=0
        while scene.object_exists(f'/Sphere/sphere{i}'):
            sphere=scene.get_object(f'/Sphere/sphere{i}')
            pc.append(sphere.get_world_pose()[0].tolist())
            i+=1

        pc=np.array(pc)
        tgt_pc=cls._tgt_pc
        
        pc=np.concatenate((pc,tgt_pc),axis=1)

        # if eval(str(cls.cfg.flow).title()):
        #     pc=np.concatenate((pc,tgt_pc-pc),axis=1) # [ 1.57756746e-01  9.57879238e-03  5.00003956e-02 -7.45579600e-04 -6.01215288e-04 -4.09781933e-07]
        # else:
        #     pc=np.concatenate((pc,tgt_pc),axis=0)     
        # # pc=np.array(rope.get_world_pose()[0]) # [np.array(pc) for pc in rope.get_world_pose()[0]],
        # if eval(str(cls.cfg.test_pc_permutation).title()) is True:
        #     pc=pc[::-1]

        eef_pos = np.array((
                Left_target_world_pos[0],
                Left_target_world_pos[1],
                Left_target_world_pos[2],
                cols[0],
                cols[1],
                cols[2],
                cols[3],
                cols[4],
                cols[5],
                gravity_dir[0],
                gravity_dir[1],
                gravity_dir[2],
                gripper_pose,
            ))

        obs = dict(
            # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
            pc=pc,
            # pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
            eef_pos=eef_pos,
            # state= eef_pos in saved npz
            # state=np.array([[Left_target_world_pos[0],Left_target_world_pos[1],Left_target_world_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])
        ) #pc and eef_pose


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
                        [o[k] for o in obs_history[-obs_horizon:]],axis=0
                    )
        for key in agent_obs.keys():
            agent_obs[key]=torch.from_numpy(np.array(agent_obs[key]))
            agent_obs[key]=agent_obs[key][None,...]
        # predict actions
        st = time.time()
        if cls.count % ac_horizon == 0:
            #print(agent_obs['pc'].shape)
            #print(type(agent_obs['pc']))

            # id= torch.eye(4, dtype=torch.float32, device='cuda')
            # rot=rpy2quat(torch.tensor([0.0,0.0,0.001],dtype=torch.float32, device='cuda'))
            # print(rot,'rot')
            # rot=kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.tensor(rot,dtype=torch.float32, device='cuda'))
            # id[:3, :3]=rot

            # id[:3, 3] = -1.0*torch.tensor([0.00001,0.00001,0.00001],dtype=torch.float32, device='cuda')*cls.count
            # ac = id[None, None, :, :].expand(1,pred_horizion,4,4)

            ac = test_batch(nets=nets, noise_scheduler=noise_scheduler,gripper_noise_scheduler=gripper_noise_scheduler, nbatch=agent_obs, device=cls.device,config=cls.config,isVisualEval=True)
            # print(ac.shape, "ac?") # b Ha 4 4
            # if eval(str(cls.cfg.manually_close).title()) is True:
            #     for i in range(len(ac)):
            #         ac[i][0]=-0.3
            cls.ac=ac.view(-1,pred_horizion,4,4).cpu()
            # print(cls.ac.shape,'predicted ac') # b Ha 4 4

        logging.info(f"Inference time: {time.time() - st:.3f}s")

        ac=cls.ac
        # take actions
        if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
            return
        agent_ac = ac[0][cls.count% ac_horizon] # if len(ac.shape) > 1 else ac    
        print("force",scene.get_object(robot_name).get_applied_action().joint_positions[-1])
        update_action(agent_ac[None,None,...],scene.get_object(target_name),scene.get_object(robot_name).end_effector,robot._gripper,eval(str(cls.cfg.rel).title()),eval(str(cls.cfg.rpy).title()),cls._sample._eps,cap=cls.cfg.cap,cup=cls.cfg.cup,update_ori=cls.cfg.update_ori,cfg=cls.cfg)
        print("force",scene.get_object(robot_name).get_applied_action().joint_positions[-1])
    
    @classmethod
    async def eval_async(cls,log_dir,reduce_horizon_dim,cfg,config,simulation_app):
        cls.log_dir=log_dir
        cls.reduce_horizon_dim=reduce_horizon_dim
        cls.config=config
        cls.cfg=cfg

        global myjson
        myjson=defaultdict(dict)
        cls.simulation_app=simulation_app
        cls.obs_horizon = config['obs_horizon']
        cls.ac_horizon = config['action_horizon']
        cls.pred_horizon = config['pred_horizon']
        cls.T_a=config['T_a']
        
        cls.device = torch.device('cuda')
        if not torch.cuda.is_available():
            cls.device = torch.device('cpu') 

        await cls._setup_async()
        cls._post_setup()
        await cls._reset_async()
        # cls._post_reset(_onDone_async=cls._reset_async)

        
def update_action(agent_ac,target,eef,gripper,rel,rpy,eps,cap=None,cup=None,update_ori=True,cfg=None):
    global myjson
    translations = agent_ac[:, :, :3, 3][0][0]

    norm = np.linalg.norm(translations)
    print(norm)
    # if norm>=0.005:
    #     translations=0.005*translations/norm
    # if norm<=0.0005:
    #     translations=0.0015*translations/norm

    rotations=agent_ac[:, :,:3, :3][0][0]
    print(rotations,translations)
    quaternions = R.from_matrix(rotations).as_quat(scalar_first=True,canonical=False) #kornia.geometry.conversions.rotation_matrix_to_quaternion(rotations)
    agent_ac=agent_ac[0][0]
    # # TODO CLIP in TRAIN + INFERENCE
    # if agent_ac[0] <0.025:
    #     gripper.close()
    # else:
    #     gripper.open()

    target_world_pos=np.array(target.get_world_pose()[0].tolist())
    target_world_ori=np.array(target.get_world_pose()[1].tolist())

    # delta_pos=np.clip(delta_pos,-0.01,0.01)
    # print("clipped delta pos: ",delta_pos)
 
    print(translations,'translations')
    print(quaternions,'quaternions')
    idx=len(myjson)
    myjson[idx]['delta_t']=translations.tolist()
    myjson[idx]['delta_q']=quaternions.tolist()
    
    tgt_pos=target_world_pos+translations.tolist()

    if tgt_pos[2]<=eps:
        print("z pos went below groundplane !!!")
        tgt_pos[2]=eps

    #tgt_ori=mu.mul(normalize_quat(np.array(quaternions.tolist())),normalize_quat(target_world_ori))
    tgt_ori=rotations@R.from_quat(target_world_ori,scalar_first=True).as_matrix()# kornia.geometry.quaternion.Quaternion(quaternions)*kornia.geometry.quaternion.Quaternion(torch.tensor(target_world_ori,dtype=torch.float32))
    
    if not update_ori:
        orientation=None
    else:
        orientation=R.from_matrix(tgt_ori).as_quat(scalar_first=True,canonical=False)

    myjson[idx]['abs_t']=tgt_pos.tolist()
    myjson[idx]['abs_q']=orientation.tolist()

    if cfg.translation is not None:
       _translation = np.array(cfg.translation)
    else:
       _translation = np.array([0,0,0])
    if cfg.rotation is not None:
        _rotation = R.from_euler('xyz', cfg.rotation, degrees=True).as_matrix()
    else:
        _rotation=np.eye(3)
    myjson[idx]['undo_abs_t']=(tgt_pos-_translation).tolist()
    gripper_world_ori= np.linalg.inv(_rotation)@R.from_quat(np.array(orientation),scalar_first=True).as_matrix()
    gripper_world_ori=R.from_matrix(gripper_world_ori).as_quat(scalar_first=True)
    myjson[idx]['undo_abs_q']=(gripper_world_ori).tolist()

    target.set_world_pose(position=tgt_pos,orientation=orientation)#.data) # tgt_ori
    print("applied pos: ",tgt_pos)
    print("applied ori: ",tgt_ori)
    
    # _gripper_status="CLOSING" if agent_ac[0] <0.025 else "opening"
    # print(f"gripper is {_gripper_status}")

# def quat_mul(q1, q2):
#     w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
#     w2, x2, y2, z2 =  q2[0], q2[1], q2[2], q2[3]
    
#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
#     return [w,x,y,z]

# def _l2_norm(q):
#     return sqrt(sum(map(lambda x: float(x)**2, q)))

# def normalize_quat(q):
#     norm=_l2_norm(q)
#     q[0],q[1],q[2],q[3]=q[0]/norm,q[1]/norm,q[2]/norm,q[3]/norm
#     return q

# def q2cols(q):
#     q=normalize_quat(q)
#     w,x,y,z=q
#     col1=[2*(w**2+x**2)-1,2*(x*y+w*z),2*(x*z-w*y)]
#     col3=[2*(w*y+x*z),2*(y*z-w*x),w**2-x**2-y**2+z**2]
#     return col1,col3

# import math

# def rpy2quat(rpy):
#     # return euler_angles_to_quat(rpy)
#     roll, pitch, yaw=rpy[0],rpy[1],rpy[2]

#     # Compute half angles
#     half_roll = roll / 2.0
#     half_pitch = pitch / 2.0
#     half_yaw = yaw / 2.0

#     # Compute trigonometric terms
#     cr = math.cos(half_roll)
#     sr = math.sin(half_roll)
#     cp = math.cos(half_pitch)
#     sp = math.sin(half_pitch)
#     cy = math.cos(half_yaw)
#     sy = math.sin(half_yaw)

#     # Compute quaternion components
#     w = cr * cp * cy + sr * sp * sy
#     x = sr * cp * cy - cr * sp * sy
#     y = cr * sp * cy + sr * cp * sy
#     z = cr * cp * sy - sr * sp * cy

#     return [w, x, y, z]

# def q2aa( q):
#     # Z is UP in isaacsim, but Y is up in dynamic control, physx and unity!
#     q = normalize_quat(q)
#     w = q[0]
#     v = np.array([q[1], q[2], q[3]])
#     angle = 2 * np.arccos(w)
#     sin_half_angle = np.sqrt(1 - w**2)
#     if angle > np.pi:
#         # angle = 2 * np.pi - angle
#         # axis = -v / sin_half_angle
#         # do not introduce discontinuity
#         # [0. 0. 1.] 3.141592653589793
#         # [-0.0029799   0.00955669 -0.99994989] 3.1302814058467474
 
#         axis = v / sin_half_angle
#     else:
#         if sin_half_angle < 1e-15:
#             axis = np.array([0.0, 0.0, 1.0])
#         else:
#             axis = v / sin_half_angle
#     return axis, angle

# def aa2q(axis, angle):
#     axis = axis / np.linalg.norm(axis)
#     half_angle = angle / 2
#     w = np.cos(half_angle)
#     xyz = axis * np.sin(half_angle)
#     # wxyz
#     return np.array([w, xyz[0], xyz[1], xyz[2]])




# def q2rmat(q):
#     q = normalize_quat(q)
#     w, x, y, z = q

#     w2 = w * w
#     x2 = x * x
#     y2 = y * y
#     z2 = z * z

#     r11 = w2 + x2 - y2 - z2
#     r12 = 2 * (x * y - w * z)
#     r13 = 2 * (x * z + w * y)

#     r21 = 2 * (x * y + w * z)
#     r22 = w2 - x2 + y2 - z2
#     r23 = 2 * (y * z - w * x)

#     r31 = 2 * (x * z - w * y)
#     r32 = 2 * (y * z + w * x)
#     r33 = w2 - x2 - y2 + z2

#     return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

