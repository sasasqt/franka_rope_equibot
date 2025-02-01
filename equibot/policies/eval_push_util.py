import omni.kit.app
from franka_pusht import ControlFlow

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

import wandb
# TODO dont block the ui: put the inference code in a new process, and cross processes communication
# TODO the objective metrics?

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
        cls.task=sample._task["Right"]
        cls.scene=world.scene
        # cls.rope=sample._rope
        cls.robot=sample._robot["Right"]
        cls.robot_name = sample._robot_name["Right"]
        cls.target_name = sample._target_name["Right"]
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
        hbar=sample._hbar
        vbar=sample._vbar
        robot=cls.robot
        robot_name=cls.robot_name
        target_name=cls.target_name
        cls.gravity_dir=[0,0,-1]


        if cls.cfg.translation is not None:
        # rotate world first after play(), otherwise the franka will compensate the rotation somehow in their code
        # rotate in simulation not in usd
            sample._world_xform.GetAttribute('xformOp:translate').Set(Gf.Vec3f(list(cls.cfg.translation))) # GetAttribute is only callable for usd objects defined via stage.DefinePrim, not for UsdGeom.Xform

        if cls.cfg.rotation is not None:
            sample._world_xform.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(list(cls.cfg.rotation)))

        if cls.cfg.scale is not None:
            sample._world_xform.GetAttribute('xformOp:scale').Set(Gf.Vec3f(list(cls.cfg.scale)))

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
                for idx,_str in enumerate(["Left","Right"]):  
                    # world.scene.get_object(robot_name).set_joint_positions(
                    #     np.array(data_frame.data[_str][f"{_str}_joint_positions"])
                    # )
                    # await omni.kit.app.get_app().next_update_async()        

                    world.scene.get_object(target_name).set_world_pose(
                        position=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                        orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                    )


                vbar.set_world_pose(
                    position=np.array(data_frame.data["T"]["vbar_world_position"]),
                    orientation=np.array(data_frame.data["T"]["vbar_world_orientation"]),
                )
                hbar.set_world_pose(
                    position=np.array(data_frame.data["T"]["hbar_world_position"]),
                    orientation=np.array(data_frame.data["T"]["hbar_world_orientation"]),
                )
            
                # rope.set_world_pose(
                #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
                # )


            right_target_world_pos=scene.get_object(target_name).get_world_pose()[0]
            right_target_world_rot=scene.get_object(target_name).get_world_pose()[1]
            col1,col3=q2cols(right_target_world_rot)
            gravity_dir=cls.gravity_dir
            if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.05 is opened
                gripper_pose=0
            else:
                gripper_pose=1

            _dict={
                "vbar_world_position": vbar.get_world_pose()[0].tolist(),
                "vbar_world_orientation": vbar.get_world_pose()[1].tolist(),
                "vbar_world_scale": vbar.get_world_scale().tolist(),
                "hbar_world_position": hbar.get_world_pose()[0].tolist(),
                "hbar_world_orientation": hbar.get_world_pose()[1].tolist(),
                "hbar_world_scale": hbar.get_world_scale().tolist(), 
            }

            _pc=[]
            values = [1, -1]
            dominant_values=np.linspace(-1, 1, num=5).tolist()
            combinations = list(product(values, repeat=2))
            combinations = [[dominant_value] + list(comb) for dominant_value in dominant_values for comb in combinations]
            for component in ["v","h"]:
                xyz=np.array(_dict[f"{component}bar_world_scale"])/2
                dominant_direction=np.argmax(xyz)
                for i,comb in enumerate(combinations):
                    tmp=comb[dominant_direction]
                    comb[dominant_direction]=comb[0]
                    comb[0]=tmp
                    center=np.array(_dict[f"{component}bar_world_position"])
                    p=np.array(comb)*xyz
                    quat_p=np.concatenate(([0.0],p))
                    ori=np.array(_dict[f"{component}bar_world_orientation"])
                    quat_p=mu.mul(mu.inverse(ori),quat_p)
                    quat_p=mu.mul(quat_p,(ori))
                    p[0],p[1],p[2]=quat_p[1],quat_p[2],quat_p[3]
                    _pc.append((center+p).tolist())

            pc=np.array(_pc)
            # NEW
            tgt_pc=np.array([[0.15260505303740501, 0.03901616483926773, -6.50063157081604e-07],
                [0.17697889357805252, 0.08267295360565186, -6.50063157081604e-07], 
                [0.11986245773732662, 0.057296548038721085, -6.50063157081604e-07], 
                [0.14423630386590958, 0.10095333773642778, -6.50063157081604e-07], 
                [0.08711986057460308, 0.07557693310081959, -6.50063157081604e-07],
                [0.11149370949715376, 0.11923372745513916, -6.50063157081604e-07], 
                [0.05437726527452469, 0.09385731909424067, -6.50063157081604e-07], 
                [0.07875111326575279, 0.1375141143798828, -6.50063157081604e-07],
                [0.021634675562381744, 0.11213770695030689, -6.50063157081604e-07], 
                [0.04600851610302925, 0.15579449757933617, -6.48200511932373e-07], 
                [0.20179031416773796, 0.1261659488081932, 1.9371509552001953e-07], 
                [0.24535439535975456, 0.10162677988409996, 1.9371509552001953e-07], 
                [0.18338593607768416, 0.09349288791418076, 1.9371509552001953e-07],
                [0.2269500158727169, 0.06895371899008751, 1.9371509552001953e-07], 
                [0.16498155891895294, 0.060819825157523155, 1.9371509552001953e-07], 
                [0.20854564011096954, 0.03628065716475248, 1.955777406692505e-07], 
                [0.14657718315720558, 0.028146762400865555, 1.9371509552001953e-07],
                [0.19014126295223832, 0.0036075934767723083, 1.955777406692505e-07],
                [0.12817280367016792, -0.004526302218437195, 1.955777406692505e-07], 
                [0.17173688486218452, -0.02906547486782074, 1.955777406692505e-07]])
            

            _dict={
                "vbar_world_position": cls._sample._target_vbar.get_world_pose()[0].tolist(),
                "vbar_world_orientation": cls._sample._target_vbar.get_world_pose()[1].tolist(),
                "vbar_world_scale": cls._sample._target_vbar.get_world_scale().tolist(),
                "hbar_world_position": cls._sample._target_hbar.get_world_pose()[0].tolist(),
                "hbar_world_orientation": cls._sample._target_hbar.get_world_pose()[1].tolist(),
                "hbar_world_scale": cls._sample._target_hbar.get_world_scale().tolist(), 
            }
            tgt_pc=[]

            values = [1, -1]
            dominant_values=np.linspace(-1, 1, num=5).tolist()
            combinations = list(product(values, repeat=2))
            combinations = [[dominant_value] + list(comb) for dominant_value in dominant_values for comb in combinations]
            for component in ["v","h"]:
                xyz=np.array(_dict[f"{component}bar_world_scale"])/2
                dominant_direction=np.argmax(xyz)
                for i,comb in enumerate(combinations):
                    tmp=comb[dominant_direction]
                    comb[dominant_direction]=comb[0]
                    comb[0]=tmp
                    center=np.array(_dict[f"{component}bar_world_position"])
                    p=np.array(comb)*xyz
                    quat_p=np.concatenate(([0.0],p))
                    ori=np.array(_dict[f"{component}bar_world_orientation"])
                    quat_p=mu.mul(mu.inverse(ori),quat_p)
                    quat_p=mu.mul(quat_p,(ori))
                    p[0],p[1],p[2]=quat_p[1],quat_p[2],quat_p[3]
                    tgt_pc.append((center+p).tolist())

            tgt_pc=np.array(tgt_pc)
            if eval(str(cls.cfg.flow).title()):
                pc=np.concatenate((pc,tgt_pc-pc),axis=1) # [ 1.57756746e-01  9.57879238e-03  5.00003956e-02 -7.45579600e-04 -6.01215288e-04 -4.09781933e-07]
            else:
                pc=np.concatenate((pc,tgt_pc),axis=0) 
            obs = dict(
                # assert isinstance(agent_obs["pc"][0][0], np.ndarray)
                pc=pc,
                # pc=np.array(rope.get_world_pose()[0]), # [np.array(pc) for pc in rope.get_world_pose()[0]],
                # state= eef_pos in saved npz
                state=np.array([[right_target_world_pos[0],right_target_world_pos[1],right_target_world_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])
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
            for idx,_str in enumerate(["Left","Right"]):   
                world.scene.get_object(robot_name).set_joint_positions(
                    np.array(data_frame.data[_str][f"{_str}_joint_positions"])
                )

            for idx,_str in enumerate(["Left","Right"]):   
                world.scene.get_object(target_name).set_world_pose(
                    position=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                    orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                )

            vbar.set_world_pose(
                position=np.array(data_frame.data["T"]["vbar_world_position"]),
                orientation=np.array(data_frame.data["T"]["vbar_world_orientation"]),
            )
            hbar.set_world_pose(
                position=np.array(data_frame.data["T"]["hbar_world_position"]),
                orientation=np.array(data_frame.data["T"]["hbar_world_orientation"]),
            )
                # rope.set_world_pose(
                #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
                # )

        # data_frame = data_logger.get_data_frame(data_frame_index=start_time-cls.obs_horizon+1+extra_repeat+i)
        # for idx,_str in enumerate(["Left","Right"]):  
        #     world.scene.get_object(robot_name).set_joint_positions(
        #         np.array(data_frame.data[_str][f"{_str}_joint_positions"])
        #     )
        cls.sample._pre_physics_callback=partial(cls._post_reset,_onDone_async=cls._reset_async)

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
        alternative_camera_prim.AddTranslateOp().Set(Gf.Vec3f([3.5,-3.9,3.5]))
        alternative_camera_prim.AddRotateXYZOp().Set(Gf.Vec3f([56,0,40]))
        # UsdGeom.Camera.Define(stage, alternative_camera_path_str)
        get_active_viewport().camera_path='/World/Camera/Camera'
        
        cls._current_time=current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.log({"output_file": current_time})
        capture_filename = "captured"
        cls._output_folder = pathlib.Path(os.getcwd()).joinpath(capture_filename)
        cls.make_sure_directory_existed(cls._output_folder)
        cls.clean_files_in_directory(cls._output_folder, ".png")

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
    def _post_reset(cls,step_size=None,_onDone_async=None):
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
        hbar=sample._hbar
        vbar=sample._vbar
        # rope=cls.rope
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

        if cls.count>cls._end:
            log_path=os.path.join(cls._output_folder, f"{cls._current_time}.json")
            print(log_path)
            cls._sample._on_save_data_event(log_path=log_path)

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
        # ISSUE 1: need at least 2 dt to populate simulation physics when gripper is holding the rope
        # ISSUE 2: during the initial alignment, the hand/gripper need to rotate to match the orientation of the taget cube
        if cls.cfg.from_demo is not None and cls.count < 0 and cls.count>=-2:
            data_frame = cls.data_logger.get_data_frame(data_frame_index=cls.start_time+1+cls.count)
            for idx,_str in enumerate(["Left","Right"]):  
                world.scene.get_object(target_name).set_world_pose(
                    position=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                    orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                )

                vbar.set_world_pose(
                    position=np.array(data_frame.data["T"]["vbar_world_position"]),
                    orientation=np.array(data_frame.data["T"]["vbar_world_orientation"]),
                )
                hbar.set_world_pose(
                    position=np.array(data_frame.data["T"]["hbar_world_position"]),
                    orientation=np.array(data_frame.data["T"]["hbar_world_orientation"]),
                )

                # rope.set_world_pose(
                #     positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                #     orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
                # )
            return

        if cls.count < 0:
            return
        
        # Make obs
        right_target_world_pos=scene.get_object(target_name).get_world_pose()[0]
        print("pos obs-ed: ", right_target_world_pos)
        right_target_world_rot=scene.get_object(target_name).get_world_pose()[1]
        col1,col3=q2cols(right_target_world_rot)
        gravity_dir=[0,0,-1]
        if scene.get_object(robot_name).get_applied_action().joint_positions[-1]< 0.025: # 0/-0.3 is closed, ~0.05 is opened
            gripper_pose=0
        else:
            gripper_pose=1

        gripper_state="CLOSED" if gripper_pose==0 else "opened"
        print(f"gripper is {gripper_state}")


        _dict={
            "vbar_world_position": vbar.get_world_pose()[0].tolist(),
            "vbar_world_orientation": vbar.get_world_pose()[1].tolist(),
            "vbar_world_scale": vbar.get_world_scale().tolist(),
            "hbar_world_position": hbar.get_world_pose()[0].tolist(),
            "hbar_world_orientation": hbar.get_world_pose()[1].tolist(),
            "hbar_world_scale": hbar.get_world_scale().tolist(), 
        }
    
        _pc=[]
        values = [1, -1]
        dominant_values=np.linspace(-1, 1, num=5).tolist()
        combinations = list(product(values, repeat=2))
        combinations = [[dominant_value] + list(comb) for dominant_value in dominant_values for comb in combinations]
        for component in ["v","h"]:
            xyz=np.array(_dict[f"{component}bar_world_scale"])/2
            dominant_direction=np.argmax(xyz)
            for i,comb in enumerate(combinations):
                tmp=comb[dominant_direction]
                comb[dominant_direction]=comb[0]
                comb[0]=tmp
                center=np.array(_dict[f"{component}bar_world_position"])
                p=np.array(comb)*xyz
                quat_p=np.concatenate(([0.0],p))
                ori=np.array(_dict[f"{component}bar_world_orientation"])
                quat_p=mu.mul(mu.inverse(ori),quat_p)
                quat_p=mu.mul(quat_p,(ori))
                p[0],p[1],p[2]=quat_p[1],quat_p[2],quat_p[3]
                _pc.append((center+p).tolist())

        pc=np.array(_pc)
        # NEW
        tgt_pc=np.array([[0.15260505303740501, 0.03901616483926773, -6.50063157081604e-07],
            [0.17697889357805252, 0.08267295360565186, -6.50063157081604e-07], 
            [0.11986245773732662, 0.057296548038721085, -6.50063157081604e-07], 
            [0.14423630386590958, 0.10095333773642778, -6.50063157081604e-07], 
            [0.08711986057460308, 0.07557693310081959, -6.50063157081604e-07],
            [0.11149370949715376, 0.11923372745513916, -6.50063157081604e-07], 
            [0.05437726527452469, 0.09385731909424067, -6.50063157081604e-07], 
            [0.07875111326575279, 0.1375141143798828, -6.50063157081604e-07],
            [0.021634675562381744, 0.11213770695030689, -6.50063157081604e-07], 
            [0.04600851610302925, 0.15579449757933617, -6.48200511932373e-07], 
            [0.20179031416773796, 0.1261659488081932, 1.9371509552001953e-07], 
            [0.24535439535975456, 0.10162677988409996, 1.9371509552001953e-07], 
            [0.18338593607768416, 0.09349288791418076, 1.9371509552001953e-07],
            [0.2269500158727169, 0.06895371899008751, 1.9371509552001953e-07], 
            [0.16498155891895294, 0.060819825157523155, 1.9371509552001953e-07], 
            [0.20854564011096954, 0.03628065716475248, 1.955777406692505e-07], 
            [0.14657718315720558, 0.028146762400865555, 1.9371509552001953e-07],
            [0.19014126295223832, 0.0036075934767723083, 1.955777406692505e-07],
            [0.12817280367016792, -0.004526302218437195, 1.955777406692505e-07], 
            [0.17173688486218452, -0.02906547486782074, 1.955777406692505e-07]])
        
        
        _dict={
            "vbar_world_position": cls._sample._target_vbar.get_world_pose()[0].tolist(),
            "vbar_world_orientation": cls._sample._target_vbar.get_world_pose()[1].tolist(),
            "vbar_world_scale": cls._sample._target_vbar.get_world_scale().tolist(),
            "hbar_world_position": cls._sample._target_hbar.get_world_pose()[0].tolist(),
            "hbar_world_orientation": cls._sample._target_hbar.get_world_pose()[1].tolist(),
            "hbar_world_scale": cls._sample._target_hbar.get_world_scale().tolist(), 
        }
        tgt_pc=[]

        values = [1, -1]
        dominant_values=np.linspace(-1, 1, num=5).tolist()
        combinations = list(product(values, repeat=2))
        combinations = [[dominant_value] + list(comb) for dominant_value in dominant_values for comb in combinations]
        for component in ["v","h"]:
            xyz=np.array(_dict[f"{component}bar_world_scale"])/2
            dominant_direction=np.argmax(xyz)
            for i,comb in enumerate(combinations):
                tmp=comb[dominant_direction]
                comb[dominant_direction]=comb[0]
                comb[0]=tmp
                center=np.array(_dict[f"{component}bar_world_position"])
                p=np.array(comb)*xyz
                quat_p=np.concatenate(([0.0],p))
                ori=np.array(_dict[f"{component}bar_world_orientation"])
                quat_p=mu.mul(mu.inverse(ori),quat_p)
                quat_p=mu.mul(quat_p,(ori))
                p[0],p[1],p[2]=quat_p[1],quat_p[2],quat_p[3]
                tgt_pc.append((center+p).tolist())
                # cls._sample._add_sphere(center+p,prim_path=f"/tgt{component}sphere{i}")
        tgt_pc=np.array(tgt_pc)
        if eval(str(cls.cfg.flow).title()):
            pc=np.concatenate((pc,tgt_pc-pc),axis=1) # [ 1.57756746e-01  9.57879238e-03  5.00003956e-02 -7.45579600e-04 -6.01215288e-04 -4.09781933e-07]
        else:
            pc=np.concatenate((pc,tgt_pc),axis=0)     
        # pc=np.array(rope.get_world_pose()[0]) # [np.array(pc) for pc in rope.get_world_pose()[0]],
        if eval(str(cls.cfg.test_pc_permutation).title()) is True:
            pc=pc[::-1]
        obs = dict(
            # assert isinstance(a"gent_obs["pc"][0][0], np.ndarray)
            pc=pc,
            # state= eef_pos in saved npz
            state=np.array([[right_target_world_pos[0],right_target_world_pos[1],right_target_world_pos[2],col1[0],col1[1],col1[2],col3[0],col3[1],col3[2],gravity_dir[0],gravity_dir[1],gravity_dir[2],gripper_pose]])
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
        if cls.count % ac_horizon == 0:

            ac = agent.act(agent_obs, return_dict=False)
            if eval(str(cls.cfg.manually_close).title()) is True:
                for i in range(len(ac)):
                    ac[i][0]=-0.3
            cls.ac=ac

        logging.info(f"Inference time: {time.time() - st:.3f}s")

        ac=cls.ac
        # take actions
        if len(obs["pc"]) == 0 or len(obs["pc"][0]) == 0:
            return
        agent_ac = ac[cls.count% ac_horizon] if len(ac.shape) > 1 else ac    
        print("force",scene.get_object(robot_name).get_applied_action().joint_positions[-1])
        update_action(agent_ac,scene.get_object(target_name),scene.get_object(robot_name).end_effector,robot._gripper,eval(str(cls.cfg.rel).title()),eval(str(cls.cfg.rpy).title()),cls._sample._eps,cap=cls.cfg.cap,cup=cls.cfg.cup,update_ori=cls.cfg.update_ori)
        print("force",scene.get_object(robot_name).get_applied_action().joint_positions[-1])

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
        # cls._post_reset(_onDone_async=cls._reset_async)

        
def update_action(agent_ac,target,eef,gripper,rel,rpy,eps,cap=None,cup=None,update_ori=True):
    # TODO CLIP in TRAIN + INFERENCE
    if agent_ac[0] <0.025:
        gripper.close()
    else:
        gripper.open()

    target_world_pos=np.array(target.get_world_pose()[0].tolist())
    target_world_ori=np.array(target.get_world_pose()[1].tolist())
    
    print("old pos: ", target_world_pos)
    print("agent pos: ",agent_ac[1:1+3])
    agent_pos=np.array(agent_ac[1:1+3])
    # delta_pos=np.clip(delta_pos,-0.01,0.01)
    # print("clipped delta pos: ",delta_pos)
 
    multiplier=1
    if rel:
        # cap fast movement
        if cap is not None:
            cap=float(cap)
            if (_l2_norm(agent_pos)*30.0>=cap): 
                # print(delta_pos,_l2_norm(delta_pos),"???")
                multiplier=30.0*_l2_norm(agent_pos)
                agent_pos=agent_pos*cap/multiplier
                # print(delta_pos,_l2_norm(delta_pos),"???")
        # cup slow movement
        if cup is not None:
            cup=float(cup)
            if (_l2_norm(agent_pos)*30.0<=cup and _l2_norm(agent_pos)*30.0>1e-8): 
                print(agent_pos,_l2_norm(agent_pos),"???")
                multiplier=30.0*_l2_norm(agent_pos)
                agent_pos=agent_pos*cup/multiplier
                print(agent_pos,_l2_norm(agent_pos),"???")
        
        tgt_pos=target_world_pos+agent_pos
    else:
        tgt_pos=agent_pos

    if tgt_pos[2]<=eps:
        print("z pos went below groundplane !!!")
        tgt_pos[2]=eps

    agent_ori=np.array(agent_ac[4:4+3])
    # delta_ori=np.clip(delta_ori,-0.05,0.05)
    angle=_l2_norm(agent_ori)
    axis=agent_ori/angle
    angle*=np.pi
    if rel:
        if rpy:
            agent_ori
            print(agent_ori)
            if agent_ori[2]<0:
                agent_ori[2]+=np.pi
                agent_ori[2]*=-1
                
            if agent_ori[2]>0:
                agent_ori[2]-=np.pi
                agent_ori[2]*=-1
            print(agent_ori)
            print()
            tgt_ori=mu.mul(normalize_quat(np.array(rpy2quat(agent_ori))),normalize_quat(target_world_ori))
        else:
            tgt_ori=mu.mul(normalize_quat(np.array(aa2q(axis,angle))),normalize_quat(target_world_ori))
    else:
        if rpy:
            tgt_ori=normalize_quat(np.array(rpy2quat(agent_ori)))
        else:
            tgt_ori=normalize_quat(np.array(aa2q(axis,angle)))
    
    print("agent ori:", agent_ori)
    print("aa ",axis,angle)
    print("tgt ori", tgt_ori)

    if not update_ori:
        tgt_ori=None
    target.set_world_pose(position=tgt_pos,orientation=tgt_ori) # tgt_ori
    print("applied pos: ",tgt_pos)
    _gripper_status="CLOSING" if agent_ac[0] <0.025 else "opening"
    print(f"gripper is {_gripper_status}")

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 =  q2[0], q2[1], q2[2], q2[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return [w,x,y,z]

def _l2_norm(q):
    return sqrt(sum(map(lambda x: float(x)**2, q)))

def normalize_quat(q):
    norm=_l2_norm(q)
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

def q2aa( q):
    # Z is UP in isaacsim, but Y is up in dynamic control, physx and unity!
    q = normalize_quat(q)
    w = q[0]
    v = np.array([q[1], q[2], q[3]])
    angle = 2 * np.arccos(w)
    sin_half_angle = np.sqrt(1 - w**2)
    if angle > np.pi:
        # angle = 2 * np.pi - angle
        # axis = -v / sin_half_angle
        # do not introduce discontinuity
        # [0. 0. 1.] 3.141592653589793
        # [-0.0029799   0.00955669 -0.99994989] 3.1302814058467474
 
        axis = v / sin_half_angle
    else:
        if sin_half_angle < 1e-15:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = v / sin_half_angle
    return axis, angle

def aa2q(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    # wxyz
    return np.array([w, xyz[0], xyz[1], xyz[2]])

