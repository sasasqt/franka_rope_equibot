# simplified in 1 file, no more extension boilerplate

# NICE TO HAVE
# TODO deformable support
# TODO numpy to torch / warp: sim and phy on GPU
# TODO proper (solver independent) force field method to randomize rope shape
# TODO use rope factory to pre-generate ropes in background in another stage
# TODO rewrite the rigidbody rope class as articulation subclass???
# TODO ~~dual robots~~, wheeled robots
# TODO a better way to align cube with gripper orientation
# TODO rewrite rope in omni.isaac.core way
# TODO use xxx view classes for high throughtput

# NEXT 
# TODO maybe to disable physics, rigidbodyapi and colliders for replay????
# TODO air drag/friction
# TODO to refactor... again...
# TODO rmpflow?
# TODO change the gripper stl! and rope thickness
# TODO rewrite isaacsimpublisher: to support openusd backend (maybe), to support visuals, to support rotations
# TODO omniverse extension on shutdown/hot reloading: cleanup 

# To launch
# TODO not ifdef vr:
# isaac only needs to bind the input once on startup
#IsaacUIUtils.setUp()

# TODO ifdef vr:
#VRUIUtils.setUp()

# the actual code
# position means coord w.r.t. global frame in omniverse
# translation means coord w.r.t. local frame in omniverse
# orientation depends on whether position or translation is used
# Z is UP in isaacsim, but Y is UP in dynamic control, physx and unity!
# unity is left handed, isaacsim is right handed
# quaternion is wxyz in isaacsim, xyzw from simpub.xr_device.meta_quest3 vr_controller.get_input_data() 
# c++ api headers are in kit\dev\fabric\include\usdrt\scenegraph\usd folder
# physx visual debugger: https://docs.omniverse.nvidia.com/extensions/latest/ext_omnipvd.html

import carb
# from viztracer import VizTracer
# tracer = VizTracer(tracer_entries=99999999,output_file="my_trace.json")

from omni.isaac.examples.base_sample import BaseSample
import numpy as np

from omni.isaac.franka import KinematicsSolver 
# from omni.isaac.franka.tasks import FollowTarget
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils import prims
from pxr import PhysxSchema

# TODO ifdef vr: import them
from simpub.sim.isaacsim_publisher import IsaacSimPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3
import carb.events
import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
# enable immediately
manager.set_extension_enabled_immediate("omni.physx.fabric", True)
from omni.physx import get_physx_interface
# physx_interface = get_physx_interface()
# num_threads = 2
# physx_interface.set_thread_count(num_threads)

from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, euler_to_rot_matrix
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction

from datetime import datetime

# world is a SimulationContext subclass
# stage uses kit app, a helper class
# scene is object subclass
# robot in articulation in single_prim_wrapper

# BaseSample is preconfigurated with phyicsScene
# methods to implement:
#     def setup_scene(cls, scene: Scene) -> None:
#     async def setup_post_load(cls):
#     async def setup_pre_reset(cls):
#     async def setup_post_reset(cls):
#     async def setup_post_clear(cls):
class FrankaRope(BaseSample):

    def _add_fixed_cylinder(self):
        from omni.isaac.core.objects import FixedCylinder
        prim = FixedCylinder(
            prim_path=find_unique_string_name(
                    initial_name=f"/World/Extras/FixedCylinder", is_unique_fn=lambda x: not is_prim_path_valid(x)
                ),
            radius=0.01,
            height=0.11,
            position=[0.0,-0.05,0.01],
            color=np.array([1.0, 0.0, 0.0])
        )
        # self.extra_prims["fixed_cylinder"]=prim

    def _add_visual_line(self):
        from omni.isaac.core.objects import VisualCylinder
        prim = VisualCylinder(
            prim_path=find_unique_string_name(
                    initial_name=f"/World/Extras/VisualCylinder", is_unique_fn=lambda x: not is_prim_path_valid(x)
                ),
            radius=0.01,
            height=1.0,
            position=[0.0,-0.05,0.05],
            orientation=[0.7071,0.0,0.7071,0.0],
            color=np.array([1.0, 0.0, 0.0])
        )

    def _add_t_shape(self):
        from omni.isaac.core.objects import DynamicCuboid
        from pxr import UsdPhysics

        stage=self._world.stage
        self._tshape_xform=_tshape_xform =stage.DefinePrim(f'/World/Extras/TShape', 'Xform')
        usd_tshape_xform = UsdGeom.Xform(_tshape_xform)
        usd_tshape_xform.AddTranslateOp().Set(Gf.Vec3f([0,0,0]))
        usd_tshape_xform.AddRotateXYZOp().Set(Gf.Vec3f([0,0,0]))
        usd_tshape_xform.AddScaleOp().Set(Gf.Vec3f([1,1,1]))

        vbar_str=find_unique_string_name(
                    initial_name=f"/World/Extras/TShape/VerticalBar", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )

        hbar_str=find_unique_string_name(
                    initial_name=f"/World/Extras/TShape/HorizontalBar", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
        
        fixed_joint = UsdPhysics.FixedJoint.Define(stage,_tshape_xform.GetPath().AppendChild("fixed_joint"))
        fixed_joint.CreateBody0Rel().SetTargets([vbar_str])
        fixed_joint.CreateBody1Rel().SetTargets([hbar_str])



        self._vbar = DynamicCuboid(
            prim_path=vbar_str,
            position=[0.0,0.0,0.01],
            color=np.array([1.0, 0.0, 0.0]),
            scale=[0.15,0.05,0.05]
        )

        self._hbar = DynamicCuboid(
            prim_path=hbar_str,
            position=[0.1,0.0,0.01],
            color=np.array([1.0, 0.0, 0.0]),
            scale=[0.05,0.15,0.05]
        )


    def _add_target_t_shape(self):
        from omni.isaac.core.objects import FixedCuboid
        from pxr import UsdPhysics

        stage=self._world.stage
        self._target_tshape_str="/World/Extras/TargetTShape"
        self._target_tshape_xform=_target_tshape_xform =stage.DefinePrim(self._target_tshape_str, 'Xform')
        usd_target_tshape_xform = UsdGeom.Xform(_target_tshape_xform)
        usd_target_tshape_xform.AddTranslateOp().Set(Gf.Vec3f([0,0,0]))
        usd_target_tshape_xform.AddRotateXYZOp().Set(Gf.Vec3f([0,0,0]))
        usd_target_tshape_xform.AddScaleOp().Set(Gf.Vec3f([1,1,1]))

        vbar_str=find_unique_string_name(
                    initial_name=f"/World/Extras/TargetTShape/VerticalBar", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )

        hbar_str=find_unique_string_name(
                    initial_name=f"/World/Extras/TargetTShape/HorizontalBar", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )

        self._target_vbar = FixedCuboid(
            prim_path=vbar_str,
            position=[0.0,0.0,-0.03],
            color=np.array([1.0, 0.0, 0.0]),
            scale=[0.15,0.05,0.05]
        )

        self._target_hbar = FixedCuboid(
            prim_path=hbar_str,
            position=[0.1,0.0,-0.03],
            color=np.array([1.0, 0.0, 0.0]),
            scale=[0.05,0.15,0.05]
        )

        # IRXR Unity does not support pure visuals for now, the workaround is to disable conlision with ground plane
        # collisionAPI = UsdPhysics.CollisionAPI.Apply(self._target_tshape_xform)
        # collisionAPI.GetCollisionEnabledAttr().Set(False)

        # collisionAPI = UsdPhysics.CollisionAPI.Apply(self._tgt_vbar.prim)
        # collisionAPI.GetCollisionEnabledAttr().Set(False)

        # collisionAPI = UsdPhysics.CollisionAPI.Apply(self._tgt_hbar.prim)
        # collisionAPI.GetCollisionEnabledAttr().Set(False)
        
    def _add_sphere(self,pos,prim_path="/sphere"):
        from omni.isaac.core.objects import VisualSphere
        sphere=VisualSphere(
            prim_path=prim_path,
            position=pos,
            color=np.array([1.0, 0.0, 0.0]),
            radius=0.01
        )

    extras={
        'fixed_cylinder': _add_fixed_cylinder,
        'visual_line': _add_visual_line,
        't_shape': _add_t_shape,
    }

    def _align_targets(self):
        for _str in ["Left","Right"]:
            # align the cube with endeffector
            position=self._robot[_str]._end_effector.get_world_pose()[0]
            self._target[_str].set_world_pose(position=position,orientation=None)
        
    def _init_vars(self):
        try:
            del self._old_observations
            del self._pre_physics_callback
            del self._post_physics_callback
        except:
            pass
        # self.extra_prims={}
        self._old_observations=None
        self._pre_physics_callback=None
        self._post_physics_callback=None

    def __init__(self,cfg=None) -> None:
        super().__init__()

        self.physics_dt=1.0/60.0
        self.rendering_dt=1.0/60.0
        self._randomize=True
        self._randomize_on_reset=True
        self.extra_scenes=None
        self._ropeLength=0.8
        self._rope_damping=0.6
        self._rope_stiffness=0.1
        self._rope_y_pos=None
        random.seed(42)

        if cfg is not None:
            print(cfg)
            self.physics_dt=eval(cfg.physics_dt)
            self.rendering_dt=eval(cfg.rendering_dt)
            self._randomize=eval(str(cfg._randomize).title())
            self._randomize_on_reset=eval(str(cfg._randomize_on_reset).title())
            self.extra_scenes=cfg.extra_scenes
            self._ropeLength=float(cfg._ropeLength)
            self._ropeLength=float(cfg._ropeLength)
            self._rope_damping=float(cfg._rope_damping)
            self._rope_stiffness=float(cfg._rope_stiffness)
            self._rope_y_pos=float(cfg._rope_y_pos) if cfg._rope_y_pos is not None else None
            random.seed(int(cfg.seed))

        self._init_vars()
        self._world_settings = {
            "physics_dt": self.physics_dt, "stage_units_in_meters": 1.0, "rendering_dt": self.rendering_dt,
            "physics_prim_path": "/PhysicsScene", "sim_params":None, "set_defaults":True, "backend":"numpy","device":None
        }
        self._eps=-0.02
        return
        
    async def setup_pre_reset(self) -> None:
        world = self._world
        self._init_vars()
        if world.physics_callback_exists("replay_recording"):
            world.remove_physics_callback("replay_recording")
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        # self._robot_articulation_solver.reset()

    async def setup_post_reset(self):
        rope=self._rope
        rope.post_reset()
        self._align_targets()
        for idx,_str in enumerate(["Left","Right"]):    
            robot=self._robot[_str]
            # close the gripper properly 
            robot._gripper.close()
            
    def world_cleanup(self):
        try:
            del self._robot_articulation_solver
        except:
            pass
        self._robot_articulation_solver = None
        return
    
    # should add tasks here
    # ~~BUT: Err set up scene:~~
    # ~~Cannot add the object target to the scene since its name is not unique~~
    # ~~slow and weird behaviour~~
    def setup_scene(self,scene=None) -> None:
        world = self._world
        world.clear()
        
        #         cylinderXform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        #         cylinderXform.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))

        # rotate world first after play(), otherwise the franka will compensate the rotation somehow in their code
        self._world_xform=_world_xform = world.stage.DefinePrim(f'/World', 'Xform')
        usd_world_xform = UsdGeom.Xform(_world_xform)
        usd_world_xform.AddTranslateOp().Set(Gf.Vec3f([0,0,0]))
        usd_world_xform.AddRotateXYZOp().Set(Gf.Vec3f([0,0,0]))
        usd_world_xform.AddScaleOp().Set(Gf.Vec3f([1,1,1]))

        # PhysX error: The application needs to increase PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity to 1101
        self._PhysicsScene=PhysicsScene = world.stage.GetPrimAtPath("/PhysicsScene")
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(PhysicsScene)
        physxSceneAPI.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(10 * 1024)
   
        # tested on amd epyc 7543p w/ 8 threads + a4500 20gb vram: cpu solver is actually faster
        # api from usdrt\scenegraph\usd\physxSchema\physxSceneAPI.h
        physxSceneAPI.CreateCollisionSystemAttr().Set("PGM")
        physxSceneAPI.CreateSolverTypeAttr().Set("TGS")
        physxSceneAPI.CreateBroadphaseTypeAttr().Set("MBP")
        # >>> Detected an articulation at /World/Franka with more than 4 velocity iterations being added to a TGS scene.The related behavior changed recently, please consult the changelog. <<<
        # AttributeError: 'Boost.Python.function' object has no attribute 'Set'
        physxSceneAPI.CreateMaxVelocityIterationCountAttr(4)
        # otherwise rendering issue in linux
        # settings = carb.settings.get_settings()
        # # async rendering
        # # w/o if statement: re-run may freeze
        # if not settings.get("/app/asyncRendering"):
        #     settings.set("/app/asyncRendering", True)
        #     settings.set("/app/asyncRendering=false",False)

        self._franka_position={}
        self._franka_orientation={}
        self._franka_inverse_position={}
        self._franka_inverse_orientation={}
        
        for idx,_str in enumerate(["Left","Right"]):
            _target_name=f"{_str}FollowedTarget"
            _target_prim_path = find_unique_string_name(
                    initial_name=f"/World/{_target_name}", is_unique_fn=lambda x: not is_prim_path_valid(x)
                )
            _target_prim = VisualCuboid(
                prim_path=_target_prim_path,
                color=np.array([1, 0, 0]),
                size=1.0,
                scale=np.array([0.02, 0.02, 0.02]) / get_stage_units()
            )
            franka_robot_name=f"{_str}_franka"
            franka_prim_path=f"/World/{franka_robot_name}"
            target_position=[0.2*idx-0.1, 0.0, 0.015]
            self._franka_position[_str]=position=[0.0,0.8*idx-0.4,0.0] # (np.linalg.inv(euler_to_rot_matrix(_world_ori)) @ np.array([0.0,0.8*idx-0.4,0.0])).tolist() 
            self._franka_inverse_position[_str]=[0.0,-0.8*idx+0.4,0.0] # (np.linalg.inv(euler_to_rot_matrix(_world_ori)) @ np.array([0.0,-0.8*idx+0.4,0.0])).tolist() 

            # rotate the left franka (left to the rope) by 180 around z: to make workspace easier
            # dont enter quaternion manually: euler_angles_to_quat for stability: eps/6.123234e-17 in [6.123234e-17 0.000000e+00 0.000000e+00 1.000000e+00]
            self._ccw=ccw=1.0
            self._franka_orientation[_str]=orientation=euler_angles_to_quat(np.array([0.0,0.0,np.pi*ccw-np.pi*ccw*idx]))
            inverse_orientation=mu.inverse(orientation)
            self._franka_inverse_orientation[_str]=inverse_orientation

            # this is the default target orientation: it aligns the target xyz axis with unrotated franka eef in world coord.
            target_orientation=euler_angles_to_quat(np.array([0,np.pi, 0])) # grab from top

            task_name=f"{_str}_follow_target_task"
            task = FollowTarget(name=task_name, target_prim_path=_target_prim_path,target_name=_target_name,target_position=target_position,target_orientation=target_orientation,franka_prim_path=franka_prim_path,franka_robot_name=franka_robot_name,
                                franka_position=position, franka_orientation=orientation,franka_gripper_closed_position=[-0.3,-0.3]
                                ) # core task api which also set_robot(), bad practice but in api # TODO
            world.add_task(task)

        # rope=self._rope=RigidBodyRope(_world=world,_ropeLength=self._ropeLength,_rope_damping=self._rope_damping,_rope_stiffness=self._rope_stiffness,_rope_y_pos=self._rope_y_pos,_randomize=self._randomize,_randomize_on_reset=self._randomize_on_reset)
        RigidBodyRope
        # try:
        #     rope.deleteRope()
        # except:
        #     pass
        # rope.createRope()

        self._add_t_shape()
        self._add_target_t_shape()
        if self.extra_scenes is not None:
            for extra in self.extra_scenes:
                self.extras[extra](self)

    # manually reset _task_scene_built to call initialize() in reset(), if task not setted up in seteup_scene
    # world._task_scene_built=False
    # await world.reset_async()

    # actually we dont need this, world.add_task() already set the world._current_tasks
    # # to mimic the load_world_async behavior from base sample.py if we setup tasks in setup_post_load
    # # since tasks are here not in setup scene
    # self._current_tasks = self._world.get_current_tasks() 

    # the followtarget task already added the groundplane
    # the basesample.py call reset_async() between setup_scene() and setup_post_load() invocations
    # reset_async() will call set_up_scene() when articulation already added to scene 
    # reset_async() will also call the initialize() method on its INITIAL call (when world._task_scene_built is False)
    # world._task_scene_built will be set to True after the initial reset_async()


    async def setup_post_load(self) -> None:
        world = self._world
        # print("rope local positions are: ",self._rope.get_local_pose()[0])
        # generate random rope shape procedurally
        # TODO play it in another stage, and copy the steady state result back into the main stage
        # better to let users do it
        # await world.play_async()
        # await asyncio.sleep(3)
        # await world.pause_async()

        # get params method in the follow target.py task returns a dict containing:
        #     ["target_prim_path"]
        #     ["target_name"]
        #     ["target_position"]
        #     ["target_orientation"]
        #     ["robot_name"]
        self._task={}
        self._target={}
        self._task_params={}
        self._robot_name={}
        self._target_name={}
        self._target_prim_path={}
        robot = self._robot={}
        self._robot_articulation_solver={}
        self._robot_articulation_controller={}
        self._data_logger  = world.get_data_logger()
        self._pre_actions={}
        self._now_actions={}
        for idx,_str in enumerate(["Left","Right"]):    
            self._pre_actions[_str]=None
            self._now_actions[_str]=None
            
            task=self._task[_str]=world.get_task(f"{_str}_follow_target_task")
            self._target[_str]=task._target
            # self._target.initialize()
            self._task_params[_str] = task_params=task.get_params() # fixed value!
            self._robot_name[_str] = robot_name=task_params["robot_name"]["value"]
            self._target_name[_str]= task_params["target_name"]["value"] 
            self._target_prim_path[_str]= task_params["target_prim_path"]["value"] 
            robot = self._robot[_str] = world.scene.get_object(robot_name)
            # it uses exts\omni.isaac.motion_generation\motion_policy_configs\franka\rmpflow\config.json
            # another way using motion policy: exts\omni.isaac.franka\omni\isaac\franka\controllers\rmpflow_controller.py
            #   motion policy: extension_examples\follow_target\follow_target.py
            self._robot_articulation_solver[_str] = KinematicsSolver(robot)
            # self._robot_articulation_solver[_str]._kinematics_solver._default_position_tolerance=0.001
            # self._robot_articulation_solver[_str]._kinematics_solver._default_orientation_tolerance=0.001
            # #self._robot_articulation_solver[_str]._kinematics._default_position_tolerance=0.001
            # #self._robot_articulation_solver[_str]._kinematics._default_orientation_tolerance=0.001
            self._robot_articulation_controller[_str] = robot.get_articulation_controller()


            # # query robot joint properties
            # >>> [Lula] Joint 'panda_finger_joint2' is specified as a mimic joint, but its control chain ['panda_finger_joint2' -> 'panda_finger_joint1'] terminates with a joint ['panda_finger_joint1'] that is not a c-space coordinate. Mimic attributes will be ignored.
            # see https://forums.developer.nvidia.com/t/isaac-2023-1-0-examples-slow-and-buggy/269980/3?u=aabb360
            # ('type', 'hasLimits', 'lower', 'upper', 'driveMode', 'maxVelocity', 'maxEffort', 'stiffness', 'damping')
            # [(0,  True, -2.8973    ,  2.8973    , 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True, -1.76279998,  1.76279998, 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True, -2.8973    ,  2.8973    , 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True, -3.07179999, -0.0698    , 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True, -2.8973    ,  2.8973    , 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True, -0.0175    ,  3.75250006, 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True, -2.8973    ,  2.8973    , 1, 5.93904701e+36, 1.00000000e+05, 5.72957824e+08, 5.729578e+06)
            # (0,  True,  0.        ,  0.04      , 1, 3.40282347e+38, 7.19999981e+00, 1.00000000e+04, 1.000000e+03)
            # (0,  True,  0.        ,  0.04      , 1, 3.40282347e+38, 3.40282347e+38, 0.00000000e+00, 0.000000e+00)]

            maxEffort = robot._articulation_view.get_max_efforts() # [[1.0000000e+05 1.0000000e+05 1.0000000e+05 1.0000000e+05 1.0000000e+05  1.0000000e+05 1.0000000e+05 7.1999998e+00 3.4028235e+38]]
            maxEffort[0,-1]=maxEffort[0,-2]
            robot._articulation_view.set_max_efforts(maxEffort)
            stiffnesses, dampings = robot._articulation_view.get_gains()
            stiffnesses[0,-1]=stiffnesses[0,-2]
            dampings[0,-1]=dampings[0,-2]
            robot._articulation_view.set_gains(kps=stiffnesses, kds=dampings)

            # print(robot.dof_properties)
            # print(robot.dof_properties.dtype.names)
            # print(robot.dof_properties)
            # print(robot.dof_names)

            # for rmpflow forward() or ik
            _robot_dof=robot.num_dof
            # In radians/s, or stage_units/s
            max_vel = np.zeros(_robot_dof) + 1.0
            max_vel[_robot_dof-1]=None # dont limit gripper
            max_vel[_robot_dof-2]=None # dont limit gripper
            robot._articulation_view.set_max_joint_velocities(max_vel)

            # # gripper open/close immediately  # dont limit gripper
            # robot._gripper._action_deltas=None

        scene=world.scene
        if scene.object_exists("default_ground_plane"):
            default_ground_plane_prim=scene.get_object("default_ground_plane")
            default_ground_plane_path=default_ground_plane_prim.prim_path
            # offset the visual part of the ground plane
            default_ground_plane_prim._xform_prim.set_default_state(position=[0,0,self._eps])
        else:
            carb.log_error(">>> default ground plane not yet added to the scene! <<<")


        robot_path={}
        for _str in ["Left","Right"]:
            robot_path[_str]=scene.get_object(self._robot_name[_str]).prim_path


        ground_plane_collision_group_path=find_unique_string_name(
                initial_name="/World/ground_plane_collision_group", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
       
        frankas_collision_group_path=find_unique_string_name(
                initial_name="/World/frankas_collision_group", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
       
        stage=world.stage
        # create collision groups
        ground_plane_collision_group = UsdPhysics.CollisionGroup.Define(stage, ground_plane_collision_group_path)
        frankas_collision_group = UsdPhysics.CollisionGroup.Define(stage, frankas_collision_group_path)
       
        # dont allow ground plane collides w/ frankas
        _filtered_rel = ground_plane_collision_group.CreateFilteredGroupsRel()
        _filtered_rel.AddTarget(Sdf.Path(frankas_collision_group_path))

        # dont allow ground plane collides w/ targets
        _filtered_rel.AddTarget(Sdf.Path(self._target_tshape_str))
        
        # ik does not consider usd obstacles
        # dont allow frankas collide w/ self, each other, and ground plane
        _filtered_rel = frankas_collision_group.CreateFilteredGroupsRel()
        _filtered_rel.AddTarget(Sdf.Path(ground_plane_collision_group_path))
        _filtered_rel.AddTarget(Sdf.Path(frankas_collision_group_path))


        # add ground plane to the ground plane group
        collectionAPI = Usd.CollectionAPI.Apply(ground_plane_collision_group.GetPrim(), "colliders")
        collectionAPI.CreateIncludesRel().AddTarget(Sdf.Path(default_ground_plane_path))


        # add frankas to franka group
        collectionAPI = Usd.CollectionAPI.Apply(frankas_collision_group.GetPrim(), "colliders")
        for _str in ["Left","Right"]:
            collectionAPI.CreateIncludesRel().AddTarget(Sdf.Path(robot_path[_str]))

        #rope_group.add_path()
        #non_rope_group.add_path()
        # otherwise no rendering in linux

        await update_stage_async()
        await world.reset_async()
        await update_stage_async()
        await self._world.pause_async()
        self._align_targets()
        for idx,_str in enumerate(["Left","Right"]):    
            robot=self._robot[_str]
            # close the gripper properly 
            robot._gripper.close()

    async def _on_replay_recording_event_async(self, data_file,callback_fn=None):
        world=self._world
        if not self._PhysicsScene.GetPrim().IsActive():
            self._PhysicsScene.GetPrim().SetActive(True)
            await world.reset_async()
            await world.pause_async()
    
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        if world.physics_callback_exists("replay_recording"):
            world.remove_physics_callback("replay_recording")
        data_logger= self._data_logger
        data_logger.load(log_path=data_file)
        await world.play_async()
        world.add_physics_callback("replay_recording", partial(self._on_replay_t_shape_recording_step,time_offset=world.current_time_step_index))
        if (callback_fn is not None):
            callback_fn()

    async def _on_simulation_event_async(self, val,callback_fn=None):
        world = self._world
        if val:
            await world.play_async()
        else:
            await world.pause_async()
        if (callback_fn is not None):
            callback_fn()
    

    async def _on_follow_target_event_async(self, val,callback_fn=None):
        world = self._world
        if val:
            # if not self._PhysicsScene.GetPrim().IsActive():
            #     self._PhysicsScene.GetPrim().SetActive(True)
            #     await world.reset_async()
            #     await omni.kit.app.get_app().next_update_async()
            if world.physics_callback_exists("replay_recording"):
                world.remove_physics_callback("replay_recording")
            if world.physics_callback_exists("sim_step"):
                world.remove_physics_callback("sim_step")
            # await world.play_async()
            world.add_physics_callback("sim_step", self._on_physics_callback)
        else:
            self._init_vars()
            if world.physics_callback_exists("sim_step"):
                world.remove_physics_callback("sim_step")
        if (callback_fn is not None):
            callback_fn()


    # in parallel gripper: 0 == close
    async def _on_gripper_action_event_async(self, name, val,callback_fn=None): 
        robot = self._robot[name]
        if val:
            robot._gripper.open() # TODO precise control?
        else:
            robot._gripper.close()
        if (callback_fn is not None):
            callback_fn()


    # "all physics callbacks shall take `step_size` as an argument"
    def _on_physics_callback(self, step_size):
        if (self._pre_physics_callback is not None):
            self._pre_physics_callback(step_size)

        self._on_follow_target_simulation_step(step_size)

        if (self._post_physics_callback is not None):
            self._post_physics_callback(step_size)

    def _on_follow_target_simulation_step(self, step_size) -> None:
        # pos, ori: via get_local_pose()
        observations =  self._world.get_observations()

        for _str in ["Left","Right"]:
            # if target stays still, do nothing
            if (self._old_observations is not None):
                _delta_pos=observations[self._target_name[_str]]["position"]-self._old_observations[self._target_name[_str]]["position"]
                # _delta_ori=observations[self._target_name[_str]]["orientation"]-self._old_observations[self._target_name[_str]]["orientation"]
                _delta_ori=mu.mul(observations[self._target_name[_str]]["orientation"],mu.inverse(self._old_observations[self._target_name[_str]]["orientation"]))
                axis,angle=_q2aa(_delta_ori)
                if (np.linalg.norm(_delta_pos)+np.linalg.norm(angle) <= np.finfo(np.dtype(_delta_pos[0])).eps):
                    continue

            # Frankas are rotated: this has to be compensated
            # 3d pos to 4d quat, correct for rotation and get the desired 3d ~~pos~~ TRANSLATION back in np.array
            #   due to this AttributeError: 'carb._carb.Float4' object has no attribute 'shape': carb to np.array
            # accounting for franka pos offset
            p1 = np.concatenate(([0.0],self._franka_inverse_position[_str]))
            p2 = np.concatenate(([0.0],observations[self._target_name[_str]]["position"]))
            _quat_pos=np.add(p1,p2)
            # accounting for franka orientation offset
            _quat_pos=mu.mul(self._franka_orientation[_str],_quat_pos)
            _quat_pos=mu.mul(_quat_pos,self._franka_inverse_orientation[_str])
            # right to left handed: xyz -> yxz coord
            # this is needed even for identity quaternion, TODO WHY?
            target_position=np.array([-_quat_pos[1],-_quat_pos[2],_quat_pos[3]])
            target_position[2]= max(self._eps, target_position[2])
            target_orientation=np.array(mu.mul(observations[self._target_name[_str]]["orientation"],self._franka_inverse_orientation[_str]))
            axis,angle=_q2aa(target_orientation)
            target_orientation=_aa2q([-axis[0],-axis[1],axis[2]],angle)

            # BUG fixed, use the correct frame
            p1 = np.concatenate(([0.0],self._franka_inverse_position[_str]))
            p2 = np.concatenate(([0.0],observations[self._target_name[_str]]["position"]))
            _quat_pos=np.add(p1,p2)
            # accounting for franka orientation offset
            _quat_pos=mu.mul(self._franka_inverse_orientation[_str],_quat_pos)
            _quat_pos=mu.mul(_quat_pos,self._franka_orientation[_str])
            # right to left handed: xyz -> yxz coord
            # this is needed even for identity quaternion, TODO WHY?
            #target_position=np.array([-_quat_pos[1],-_quat_pos[2],_quat_pos[3]])
            target_position[2]= max(self._eps, target_position[2])
            target_orientation=np.array(mu.mul(self._franka_orientation[_str],observations[self._target_name[_str]]["orientation"]))


            # compute_inverse_kinematics does not expect carb._carb.Float4 as inputs
            # target_position (np.array): target **translation** of the target frame (in stage units) relative to the USD stage origin
            # target_orientation (np.array): target orientation of the target frame relative to the USD stage global frame. Defaults to None.
            actions, succ = self._robot_articulation_solver[_str].compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
            )

            # HELL NO (succ == True) != successed
            # even early return in case of NaN will still cause the BroadPhaseUpdateData error ->  should clip motion.
            #   done via max velocities
            # None in action is normal, nan is not
            # >>> old tgt [-0.10554442  0.29619464  0.015     ] <<<
            # >>> new tgt [-0.12670761  0.3028575   0.015     ] <<<
            # actions {'joint_positions': [nan, nan, nan, nan, nan, nan, nan], 'joint_velocities': None, 'joint_efforts': None} is True
            if succ:

                self._pre_actions[_str]=self._now_actions[_str] or actions.get_dict()['joint_positions']
                self._now_actions[_str] = actions.get_dict()['joint_positions']
                _norm=np.linalg.norm(np.array(self._now_actions[_str])-np.array(self._pre_actions[_str]))

                if (_norm>0.5):
                    # TELEPORTING: this is easier than to correct the eef velocity afterwards (gripper vel cannot be set directly)
                    # ValueError: shape mismatch: value array of shape (1,7) could not be broadcast to indexing result of shape (1,9)
                    # the last 2 are left right finger
                    self._robot[_str].set_joint_positions(self._now_actions[_str],joint_indices=np.arange(7)) 
                else:
                    self._robot_articulation_controller[_str].apply_action(actions)

            else:
                carb.log_warn("IK did not converge to a solution. No action is being taken.")

        self._old_observations=observations

    def _on_replay_t_shape_recording_step(self, step_size, time_offset=0):
        world=self._world
        data_logger=self._data_logger
        vbar=self._vbar
        hbar=self._hbar
        if world.current_time_step_index-time_offset < data_logger.get_num_of_data_frames():
            if self._PhysicsScene.GetPrim().IsActive():
                # disable the PhysicsScene to for fps
                self._PhysicsScene.GetPrim().SetActive(False)
            for idx,_str in enumerate(["Left","Right"]):  
                robot_name = self._robot_name[_str]
                target_name = self._target_name[_str]
                data_frame = data_logger.get_data_frame(data_frame_index=world.current_time_step_index-time_offset)
                if idx == 0:
                    world.scene.get_object(robot_name).set_joint_positions(
                        np.array(data_frame.data[_str][f"{_str}_joint_positions"])
                    )
                else:
                    world.scene.get_object(robot_name).apply_action(ArticulationAction(joint_positions=np.array(data_frame.data[_str]["applied_joint_positions"])))
                
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
            
            
        else:
            asyncio.ensure_future(world.pause_async())
            # if not self._PhysicsScene.GetPrim().IsActive():
            #     self._PhysicsScene.GetPrim().SetActive(True)
            #     async def _reset_async(world):
            #         await world.reset_async()
            #         await world.pause_async()
            #     asyncio.ensure_future(_reset_async(world))
            # else:
            #     asyncio.ensure_future(world.pause_async())
            # asyncio.ensure_future(world.pause_async())
            if world.physics_callback_exists("replay_recording"):
                world.remove_physics_callback("replay_recording")
        return
    
    def _on_replay_rope_recording_step(self, step_size, time_offset=0):
        world=self._world
        data_logger=self._data_logger
        rope=self._rope
        if world.current_time_step_index-time_offset < data_logger.get_num_of_data_frames():
            if self._PhysicsScene.GetPrim().IsActive():
                # disable the PhysicsScene to for fps
                self._PhysicsScene.GetPrim().SetActive(False)
            for idx,_str in enumerate(["Left","Right"]):  
                robot_name = self._robot_name[_str]
                target_name = self._target_name[_str]
                data_frame = data_logger.get_data_frame(data_frame_index=world.current_time_step_index-time_offset)
                if idx == 0:
                    world.scene.get_object(robot_name).set_joint_positions(
                        np.array(data_frame.data[_str][f"{_str}_joint_positions"])
                    )
                else:
                    world.scene.get_object(robot_name).apply_action(ArticulationAction(joint_positions=np.array(data_frame.data[_str]["applied_joint_positions"])))
                
                world.scene.get_object(target_name).set_world_pose(
                    position=np.array(data_frame.data[_str][f"{_str}_target_world_position"]),
                    orientation=np.array(data_frame.data[_str][f"{_str}_target_world_orientation"])
                )
            rope.set_world_pose(
                positions=np.array(data_frame.data["Rope"]["Rope_world_position"]),
                orientations=np.array(data_frame.data["Rope"]["Rope_world_orientation"]),
            )
        else:
            asyncio.ensure_future(world.pause_async())
            # if not self._PhysicsScene.GetPrim().IsActive():
            #     self._PhysicsScene.GetPrim().SetActive(True)
            #     async def _reset_async(world):
            #         await world.reset_async()
            #         await world.pause_async()
            #     asyncio.ensure_future(_reset_async(world))
            # else:
            #     asyncio.ensure_future(world.pause_async())
            # asyncio.ensure_future(world.pause_async())
            if world.physics_callback_exists("replay_recording"):
                world.remove_physics_callback("replay_recording")
        return
    
    def _on_logging_event(self, val,callback_fn=None,extras_fn=None):
        world = self._world
        
        self.data_logger = data_logger = world.get_data_logger()
        if not data_logger.is_started():


            # rope=self._rope
            def frame_logging_func(tasks, scene):
                _dict={}
                for _str in ["Left","Right"]:
                    robot_name = self._robot_name[_str]
                    target_name = self._target_name[_str]
                    _dict[_str]= {
                        f"{_str}_joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),
                        "applied_joint_positions": scene.get_object(robot_name)
                            .get_applied_action()
                            .joint_positions.tolist(),
                        f"{_str}_end_effector_world_position":scene.get_object(robot_name).end_effector.get_world_pose()[0].tolist(),
                        f"{_str}_end_effector_world_orientation":scene.get_object(robot_name).end_effector.get_world_pose()[1].tolist(),
                        f"{_str}_end_effector_local_position":scene.get_object(robot_name).end_effector.get_local_pose()[0].tolist(),
                        f"{_str}_end_effector_local_orientation":scene.get_object(robot_name).end_effector.get_local_pose()[1].tolist(),
                        
                        # f"{_str}_target_local_position": scene.get_object(target_name).get_local_pose()[0].tolist(),
                        # f"{_str}_target_local_orientation": scene.get_object(target_name).get_local_pose()[1].tolist(),
                        f"{_str}_target_world_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
                        f"{_str}_target_world_orientation": scene.get_object(target_name).get_world_pose()[1].tolist(),
                        f"{_str}_target_local_position": scene.get_object(target_name).get_local_pose()[0].tolist(),
                        f"{_str}_target_local_orientation": scene.get_object(target_name).get_local_pose()[1].tolist(),
                    }
                # _dict["Rope"]={ 
                #         # ~~orientation data is too large and unnecessary~~ it makes replay easier
                #         "Rope_world_position": rope.get_world_pose()[0], #rope.get_world_position(),
                #         "Rope_world_orientation": rope.get_world_pose()[1],
                #     }
                _dict["T"]={
                    "vbar_world_position": self._vbar.get_world_pose()[0].tolist(),
                    "vbar_world_orientation": self._vbar.get_world_pose()[1].tolist(),
                    "vbar_scale": self._vbar.get_local_scale().tolist(),
                    "hbar_world_position": self._hbar.get_world_pose()[0].tolist(),
                    "hbar_world_orientation": self._hbar.get_world_pose()[1].tolist(),
                    "hbar_scale": self._vbar.get_local_scale().tolist(), 
                }


                dict=_dict["T"]={
                    "vbar_world_position": self._vbar.get_world_pose()[0].tolist(),
                    "vbar_world_orientation": self._vbar.get_world_pose()[1].tolist(),
                    "vbar_world_scale": self._vbar.get_world_scale().tolist(),
                    "hbar_world_position": self._hbar.get_world_pose()[0].tolist(),
                    "hbar_world_orientation": self._hbar.get_world_pose()[1].tolist(),
                    "hbar_world_scale": self._hbar.get_world_scale().tolist(), 
                }
                pc=[]
                from itertools import product
                values = [1, -1]
                dominant_values=np.linspace(-1, 1, num=5).tolist()
                combinations = list(product(values, repeat=2))
                combinations = [[dominant_value] + list(comb) for dominant_value in dominant_values for comb in combinations]
                for component in ["v","h"]:
                    xyz=np.array(dict[f"{component}bar_world_scale"])/2
                    dominant_direction=np.argmax(xyz)
                    for i,comb in enumerate(combinations):
                        tmp=comb[dominant_direction]
                        comb[dominant_direction]=comb[0]
                        comb[0]=tmp
                        center=np.array(dict[f"{component}bar_world_position"])
                        p=np.array(comb)*xyz
                        quat_p=np.concatenate(([0.0],p))
                        ori=np.array(dict[f"{component}bar_world_orientation"])
                        quat_p=mu.mul(mu.inverse(ori),quat_p)
                        quat_p=mu.mul(quat_p,(ori))
                        p[0],p[1],p[2]=quat_p[1],quat_p[2],quat_p[3]
                        pc.append((center+p).tolist())
                        # self._add_sphere(center+p,prim_path=f"/{component}sphere{i}")
                _dict["T"]["pc"]=pc

                if extras_fn is not None:
                    _dict["extras"]=extras_fn()
                _dict["Datetime"]={"now":datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}
                return _dict

            data_logger.add_data_frame_logging_func(frame_logging_func)
            
        if val:
            data_logger.start()
        else:
            data_logger.pause()
        if (callback_fn is not None):
            callback_fn()

    def _on_save_data_event(self, log_path=None,callback_fn=None):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if log_path is None:
            log_path=os.path.join(os.getcwd(), f"output_data_{current_time}.json")
        print(f">>> log path: {log_path}")
        world = self._world
        data_logger = self.data_logger=world.get_data_logger()
        data_logger.save(log_path=log_path)
        data_logger.reset()
        if (callback_fn is not None):
            callback_fn()


def _q2aa( q):
    # Z is UP in isaacsim, but Y is up in dynamic control, physx and unity!
    q = q / np.linalg.norm(q)
    w = q[0]
    v = np.array([q[1], q[2], q[3]])
    angle = 2 * np.arccos(w)
    sin_half_angle = np.sqrt(1 - w**2)
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -v / sin_half_angle
    else:
        if sin_half_angle < 1e-15:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = v / sin_half_angle
    return axis, angle

def _aa2q(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    # wxyz
    return np.array([w, xyz[0], xyz[1], xyz[2]])


# the async ext utils for stage hot reload etc
import asyncio
from omni.isaac.core.utils.stage import (
    create_new_stage_async,
    is_stage_loading,
    update_stage_async,
    clear_stage,
    close_stage,
    create_new_stage,
)

# ~~anti feature/no impl: ui~~ impl below for both button binding in isaac and vr

import os
import omni.ui
from omni.isaac.ui.ui_utils import btn_builder, state_btn_builder, str_builder

import carb.input
import omni.appwindow

from typing import Optional, Dict, List, Any
from dataclasses import dataclass

class Button:
    default_state: str
    alternative_state: Optional[str] = None
    current_state: str

    def __init__(self,default_state,alternative_state,current_state):
        self.default_state=default_state
        self.alternative_state=alternative_state
        self.current_state=current_state

    def get_state_and_flip(self):
        started=self.current_state == self.default_state
        self.get_or_set_state(started)
        return started
    
    def get_or_set_state(self, toStart:bool=None):
        if toStart is None:
            return self.current_state == self.default_state
        if not toStart:
            self.default()
        else:
            self.alternative()
    

    def default(self):
        self.current_state=self.default_state

    def alternative(self):
        self.current_state=self.alternative_state


class ControlFlow:
    _sample=None        
    buttons: Optional[Dict[str, Button]] = None
    BUTTON_CONFIG: Dict[str, List[str]] = {
        "(Re)load": ["(re)load"],
        "Stop(Reset)": ["stop(reset)"],
        "Clear": ["clear"],
        "Follow Target": ["to start", "to stop"],
        "Simulation": ["to play", "to pause"],
        "Left Gripper Action": ["to open", "to close"],
        "Right Gripper Action": ["to open", "to close"],
        "Start Logging": ["to begin", "to stop"],
        "Save Data": ["save"],
        "Replay Recording": ["replay"],
    }

    @classmethod
    def _populate_button(cls, name: str, state: str, alt_state: Optional[str] = None) -> None:
        cls.buttons[name] = Button(state, alt_state, state)

    @classmethod
    def reset_buttons(cls,callback_fn=None):
        for name,button in cls.buttons.items():
            cls.buttons[name].default()
        if (callback_fn is not None):
            callback_fn()

    @classmethod
    def init_buttons(cls,callback_fn=None) -> None:
        if cls.buttons is not None:
            cls.reset_buttons()
            return
        cls.buttons={}
        try:
            for name, states in cls.BUTTON_CONFIG.items():
                alt_state = states[1] if len(states) > 1 else None
                cls._populate_button(name, states[0], alt_state)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize buttons: {str(e)}")
        if (callback_fn is not None):
            callback_fn()

    @classmethod
    async def setUp_async(cls,cfg=None, callback_fn=None):
        if (cls._sample is not None):
            try:
                await cls.tearDown_async()
            except:
                pass
        # await create_new_stage_async()
        # await update_stage_async()
        cls._sample = FrankaRope(cfg) # TODO replace with yaml
        await update_stage_async()
        await cls._sample.load_world_async()
        await update_stage_async()
        if (callback_fn is not None):
            callback_fn()
    

    @classmethod
    async def tearDown_async(cls,callback_fn=None) -> None:
        if (cls._sample is not None):
            await cls._sample.clear_async() # clear async does not close stage
        await update_stage_async()
        try:
            del cls._sample
        except:
            pass
        cls._sample = None
        if (callback_fn is not None):
            callback_fn()
            
    @classmethod
    async def reset_async(cls,callback_fn=None) -> None:
        if (cls._sample is not None):
            await cls._sample.reset_async()
        if callback_fn is not None:
            callback_fn()
        await update_stage_async()

    @classmethod
    def on_reload(cls,callback_fn=None):
        async def _on_reload_async(callback_fn=None):
            cls.init_buttons()
            await cls.setUp_async(callback_fn)
        asyncio.ensure_future(_on_reload_async(callback_fn))

    @classmethod
    def on_reset(cls,callback_fn=None):
        async def _on_reset_async(callback_fn=None):
            cls.init_buttons()
            await cls.reset_async(callback_fn)
        asyncio.ensure_future(_on_reset_async(callback_fn))
    
    @classmethod
    def on_clear(cls,callback_fn=None):
        async def _on_clear_async(callback_fn=None):
            cls.init_buttons()
            await cls.tearDown_async(callback_fn)
        asyncio.ensure_future(_on_clear_async(callback_fn))

    @classmethod
    def on_simulation_button_event(cls, playing=None,callback_fn=None):
        async def _on_simulation_button_event_async(playing=None,callback_fn=None):
            val = cls.buttons["Simulation"].get_state_and_flip()
            if playing is None:
                playing=val
            else:
                cls.buttons["Simulation"].get_or_set_state(playing)
            # two seprate states: playing and val: workaround for isaac sim's stateful a/b-text buttons
            # val is the internal state maintained by button class
            # playing is the external state maintained by isaac sim's state buttons
            assert (playing == val)
            if not playing:
                pass
                # better: not modifty the task state
                # cls.buttons["Follow Target"].default()
                # await cls._sample._on_follow_target_event_async(False)
                #tracer.stop()
                #tracer.save()
            else:
                #tracer.start()
                pass
            await cls._sample._on_simulation_event_async(playing,callback_fn)
        asyncio.ensure_future(_on_simulation_button_event_async(playing,callback_fn))
    
    @classmethod # TODO maybe refactor using publisher subscriber/callback
    def on_follow_target_button_event(cls, following=None,callback_fn=None):
        async def _on_follow_target_button_event_async(following=None,callback_fn=None):
            val=cls.buttons["Follow Target"].get_state_and_flip()
            if following is None:
                following=val
            else:
                cls.buttons["Follow Target"].get_or_set_state(following)
            assert (following == val)
            if following:
                cls.buttons["Simulation"].alternative()
            else:
                cls.buttons["Simulation"].default()
            await cls._sample._on_follow_target_event_async(following,callback_fn)
        asyncio.ensure_future(_on_follow_target_button_event_async(following,callback_fn))

    @classmethod
    def on_gripper_action_button_event(cls, name, open=None,callback_fn=None):
        async def _on_gripper_action_button_event_async(cls,name,open=None,callback_fn=None):
            # print(f"name {name}")
            
            val=cls.buttons[f"{name} Gripper Action"].get_state_and_flip()
            # print(f"val {val}")
            # print(f"open {open}")
            if open is None:
                open=val
            else:
                cls.buttons[f"{name} Gripper Action"].get_or_set_state(open)
            assert (open == val)
            await cls._sample._on_gripper_action_event_async(name,open,callback_fn)
        asyncio.ensure_future(_on_gripper_action_button_event_async(cls,name,open,callback_fn))

    @classmethod
    def on_logging_button_event(cls, logging=None,callback_fn=None,extras_fn=None):
        val=cls.buttons["Start Logging"].get_state_and_flip()
        if logging is None:
            logging=val
        else:
            cls.buttons["Start Logging"].get_or_set_state(logging)
        assert (logging == val)
        cls._sample._on_logging_event(logging,callback_fn,extras_fn)

    @classmethod
    def on_save_data_button_event(cls,callback_fn=None):
        cls._sample._on_save_data_event(callback_fn)

    @classmethod
    def on_replay_recording_button_event(cls,datafile,callback_fn=None):
        asyncio.ensure_future(cls._sample._on_replay_recording_event_async(datafile,callback_fn))

# singleton
# all UI related
from functools import partial
class IsaacUIUtils(ControlFlow):
    isaac_buttons=None
    ui_window=None

    @classmethod
    def bind_inputs(cls):
        if os.getenv("__input_bound") is not None:
            return
        app_window = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = app_window.get_keyboard()
        cls.sub_id = input_interface.subscribe_to_keyboard_events(keyboard, cls.on_input)
        os.environ["__input_bound"]=str(cls.sub_id)

    @classmethod
    def unbind_inputs(cls):
        if os.getenv("__input_bound") is None:
            return
        sub_id=cls.sub_id or int(os.getenv("__input_bound"))
        app_window = omni.appwindow.get_default_app_window()
        input_interface = carb.input.acquire_input_interface()
        keyboard = app_window.get_keyboard()
        input_interface.unsubscribe_to_keyboard_events(keyboard,sub_id)

    @classmethod
    def on_input(cls,event):
        if event.input == carb.input.KeyboardInput.SPACE:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if not cls._sample._world.is_playing():
                    cls.on_simulation_button_event(True)
                else:
                    cls.on_simulation_button_event(False)
        if event.input == carb.input.KeyboardInput.J:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                super().on_gripper_action_button_event(name="Left", callback_fn=cls._update_ui_button_text)
        if event.input == carb.input.KeyboardInput.K:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                super().on_gripper_action_button_event(name="Right", callback_fn=cls._update_ui_button_text)


    @classmethod
    def setUp(cls, cfg=None,window_name: str = "Franka Rope") -> None:
        cls.cfg=cfg
        print("Creating window for environment.")
        ui_window = omni.ui.Workspace.get_window(window_name) or cls.ui_window
        if (ui_window is not None):
            # cls.tearDown()
            return # dont create ui window on hot reloading, also dont destroy exsiting window either
        async def _setUp_async(cls):
            await omni.kit.app.get_app().next_update_async()
            await super().setUp_async(cls.cfg)
            # cls.bind_inputs()
            cls.ui_window = omni.ui.Window(window_name, width=300, height=300)
            cls.ui_window.flags = (omni.ui.WINDOW_FLAGS_NO_CLOSE)
            cls.add_buttons()
        asyncio.ensure_future(_setUp_async(cls))
        return
    
    @classmethod
    def tearDown(cls,window_name: str = "Franka Rope") -> None:
        asyncio.ensure_future(super().tearDown_async())
        # destroy the window
        try: # also works with no input bound
            cls.unbind_inputs()
        except:
            pass
        ui_window = omni.ui.Workspace.get_window(window_name) or cls.ui_window # just like x = a || b 
        if ui_window is not None:
            ui_window.visible = False
            ui_window.destroy()
            ui_window = None
        return

    @classmethod
    def _update_ui_button_text(cls):
        for name,dict in cls.isaac_buttons.items():
            if "b_text" in dict:
                cls.isaac_buttons[name]["ui_button"].text=cls.buttons[name].current_state.upper() # omniverse buttons are in CAP
        cls._enable_ui_button_text()

    @classmethod
    def _enable_ui_button_text(cls, to_enable=None):
        if to_enable is None:
            to_enable=cls.isaac_buttons.keys()
        for name in to_enable:
            cls.isaac_buttons[name]["ui_button"].enabled=True
        for name,dict in cls.isaac_buttons.items():
            if name not in to_enable:
                cls.isaac_buttons[name]["ui_button"].enabled=False
        
    @classmethod
    def build_buttons(cls):
        def _on_replay_recording_button_event():
            to_enable=["(Re)load", "Replay Recording"]
            cls._enable_ui_button_text(to_enable=to_enable)
            cls.on_replay_recording_button_event(datafile=cls._Recording_To_Replay.get_value_as_string())
    
        _callback_fn=cls._update_ui_button_text
        _on_reload=partial(super().on_reload,callback_fn=_callback_fn)
        _on_reset=partial(super().on_reset,callback_fn=_callback_fn)
        _on_clear=partial(super().on_clear,callback_fn=_callback_fn)
        _on_follow_target_button_event=partial(super().on_follow_target_button_event,callback_fn=_callback_fn)
        _on_simulation_button_event=partial(super().on_simulation_button_event,callback_fn=_callback_fn)
        _on_left_gripper_action_button_event=lambda open:cls.on_gripper_action_button_event(name="Left",open=open, callback_fn=_callback_fn) # partial(super().on_gripper_action_button_event,name="Left",callback_fn=_callback_fn)
        _on_right_gripper_action_button_event=lambda open:cls.on_gripper_action_button_event(name="Right",open=open, callback_fn=_callback_fn) # partial(super().on_gripper_action_button_event,name="Right",callback_fn=_callback_fn)
        _on_logging_button_event=partial(super().on_logging_button_event,callback_fn=_callback_fn)

        cls.FN_CONFIG={
            "(Re)load": _on_reload,
            "Stop(Reset)": _on_reset,
            "Clear": _on_clear,
            "Follow Target": _on_follow_target_button_event,
            "Simulation": _on_simulation_button_event,
            "Left Gripper Action": _on_left_gripper_action_button_event,
            "Right Gripper Action": _on_right_gripper_action_button_event,
            "Start Logging": _on_logging_button_event,
            "Save Data": super().on_save_data_button_event,
            "Replay Recording":_on_replay_recording_button_event,
        }
        
        dicts={}
        for name, button in cls.buttons.items():
            dicts[name]={
                "label":name,
                "type":"button",
            }
            if button.alternative_state is None:
                dicts[name]["text"]=name
            else:
                dicts[name]["a_text"]=button.default_state
                dicts[name]["b_text"]=button.alternative_state
        for name, fn in cls.FN_CONFIG.items():
            dicts[name]["on_clicked_fn"]=fn
        
        cls.isaac_buttons=dicts

    @classmethod
    def add_buttons(cls):
        if cls.buttons is None: # super().param != cls.param
            super().init_buttons()
        if cls.isaac_buttons is not None:
            return
        cls.build_buttons()
        with cls.ui_window.frame:
            with omni.ui.VStack():
                for name,dict in cls.isaac_buttons.items():
                    if "b_text" in dict:
                        _btn=state_btn_builder(**dict)
                        _btn.enabled=True
                    else:
                        _btn=btn_builder(**dict)
                    cls.isaac_buttons[name]["ui_button"]=_btn

                dict = {
                    "label": "Recording To Replay",
                    "type": "stringfield",
                    "default_val": os.path.join(os.getcwd(), "SAVED_DATA.json"),
                    "tooltip": "Output Directory",
                    "on_clicked_fn": None,
                    "use_folder_picker": True,
                    "read_only": False,
                }
                cls._Recording_To_Replay=str_builder(**dict)




# semi hidden api
# https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.utils/docs/index.html#omni.isaac.utils._isaac_utils.math.mul
from omni.isaac.utils._isaac_utils import math as mu
from concurrent.futures import ThreadPoolExecutor
import threading
import time
lock=threading.Lock()

class VRUIUtils(ControlFlow):

    @classmethod
    def _init_vars(cls):
        try:
            del cls.publisher
            del cls.vr_controller
            del cls.old_input_pos
            del cls.old_input_rot
            del cls.old_input_data
            del cls.input_data
            del cls.already_pressed
            del cls.names
            del cls._zeroing
        except:
            pass
        cls.publisher=None
        cls.vr_controller=None
        cls.old_input_pos=None
        cls.old_input_rot=None
        cls.old_input_data=None
        cls.input_data=None
        cls.already_pressed={}
        cls.names=[]
        cls._zeroing=False

    @classmethod
    def init_publisher(cls,vr_controller=None):
        def _init_publisher(cls,vr_controller=None):
            world=cls._sample._world
            if (cls.publisher is None):
                print(">>> INIT SIMPUBLISHER <<< ")
                cls.publisher = IsaacSimPublisher(host="192.168.0.103", stage=world.stage) # for InteractiveScene
            # THE MetaQuest3 NAME MUST BE THE SAME AS IN THE CSHARP CODE
            if (cls.vr_controller is None):
                print(">>> INIT META QUEST 3 <<< ")
                # MetaQuest3 only r w/ isaac sim fabric backend
                cls.vr_controller=vr_controller or MetaQuest3("UnityClient")#MetaQuest3("ALRMetaQuest3")
                # print(f"world is playing: {world.is_playing()}")
            super().on_simulation_button_event(False)
        super().on_simulation_button_event(True,callback_fn=partial(_init_publisher,cls,vr_controller))

    @classmethod
    async def init_publisher_async(cls,vr_controller=None):
        cls.init_publisher(vr_controller)
    
    @classmethod
    def destroy_publisher(cls):
        if cls.publisher is not None:
            cls.publisher.scene_update_streamer.shutdown()
            cls.publisher.scene_service.shutdown()
            cls.publisher.asset_service.shutdown()
            try:
                del cls.publisher
            except:
                pass
            cls.publisher=None
        # if cls.vr_controller is not None:

    @classmethod
    async def destroy_publisher_async(cls):
        cls.destroy_publisher()

    @classmethod
    def setUp(cls,cfg=None,vr_controller=None):
        cls.cfg=cfg
        cls._init_vars()
        for name,_ in cls.BUTTON_CONFIG.items():
            cls.names.append(name)
        cls.names.append("Reposition")

        async def _setUp_async(cls,vr_controller=None):
            super().init_buttons()

            cls.reset_press_states()
            if (cls.vr_controller is not None):
                cls.tearDown()
                
            await super().setUp_async(cls.cfg)

            # issue: isaacsim/omniverse on init launch does not populate rt_prim without out pressing simulation start first (register the timeline event)
            # issue: w/o timeline play, the vr controller wont be registered to send back data
            #   fixed in init_publisher method via callbacks
            cls.init_publisher()

            loop = asyncio.get_event_loop()
            await asyncio.to_thread(partial(cls.ui_daemon,loop))
        asyncio.ensure_future(_setUp_async(cls,vr_controller))

    # an async listening demon running in another thread
    @classmethod
    def ui_daemon(cls,loop):
        asyncio.set_event_loop(loop)
        while True:
            start = time.time()
            try:
                cls.bind_inputs()
            except:
             pass
            end = time.time()
            if (end-start<0.009): # TODO maybe not hard coding the pulling time
                time.sleep(0.009-(end-start))

    @classmethod
    def pre_physics_callback(cls,step_size):
        with lock:
            try:
                if cls._requested_zeroing_pose():
                    cls.register_zeroing_pose()
                    cls._zeroing=False
            except:
                # the input is none, 
                pass
        cls.default_task()

    @classmethod
    def _requested_zeroing_pose(cls):
        return cls._zeroing
            
    @classmethod
    def _request_zeroing_pose(cls):
        cls._zeroing=True

    @classmethod
    def register_zeroing_pose(cls):
        input_data=cls._get_input_data() # vr_controller.get_input_data()        
        cls.zeroing_pose(input_data)

    # TODO this is not elegant! 
    # either isaac simpublisher or irxr unity rotates the xy plane by 180 degrees, undo that rotation here
    @classmethod
    def _get_input_data(cls):
        input_data={}
        _input_data=cls.vr_controller.get_input_data()
        input_data['left']=_input_data['right']
        input_data['right']=_input_data['left']
        input_data['X']=_input_data['X']
        input_data['Y']=_input_data['Y']
        input_data['A']=_input_data['A']
        input_data['B']=_input_data['B']
        return input_data
    
    @classmethod
    def default_task(cls):
        input_data=cls._get_input_data() # vr_controller.get_input_data()
        # print(input_data)
        cls.transform_target(input_data)
        cls.reset_press_states()
        return

    @classmethod
    def reset_press_states(cls):
        for name in cls.names:
            cls.already_pressed[name]=False

    @classmethod
    def bind_inputs(cls):

        # control logic: 
        #   each button can be pressed max once between each simulation step
        # control logic priority:
        # 1. return if the input is invalid
        # 2. return on reset
        # 3. return on simulation Simulation
        # 4. return on follow target task start/stop
        # 5. continue only if **none** of the above conditions meet: update pose and gripper action
        if cls.vr_controller is None:
            return
        cls.old_input_data=cls.input_data or cls._get_input_data() # vr_controller.get_input_data()
        input_data = cls.input_data= cls._get_input_data() # vr_controller.get_input_data()
        if input_data is None:
            return
        if cls._sample is None:
            return
        
        if cls._reset_pressed(input_data) and not cls.already_pressed["Stop(Reset)"]:
            cls.reset_press_states()
            try:
                cls._sample.data_logger.reset()
            except:
                pass
            cls.already_pressed["Stop(Reset)"]=True
            cls._request_zeroing_pose()
            _onFinish=lambda: print("vibrate") # vibrate # TODO 
            def _onFinish(cls):
                async def _onFinish_async(cls):
                    super().on_simulation_button_event(True,callback_fn=partial(super().on_simulation_button_event,False))
                asyncio.ensure_future(_onFinish_async(cls))
            super().on_reset(_onFinish(cls))
            return
        
        if cls._simulation_pressed(input_data) and not cls.already_pressed["Simulation"]:
            cls.reset_press_states()
            cls.already_pressed["Simulation"]=True
            val=cls.buttons["Simulation"].get_or_set_state()
            if val:
                cls._request_zeroing_pose()                
            super().on_simulation_button_event(val,cls.reset_press_states)
            return
        
        if cls._follow_target_pressed(input_data) and not cls.already_pressed["Follow Target"]:
            cls.already_pressed["Follow Target"]=True
            val=cls.buttons["Follow Target"].get_or_set_state()
            if val:
                super().on_logging_button_event(True,extras_fn=cls._extras_logging)
                cls._request_zeroing_pose()
            else:
                super().on_save_data_button_event()
                pass
            def _pre_physics_callback(val):
                sample = cls._sample
                if not val:
                    try: 
                        del cls._sample._pre_physics_callback
                        cls._sample._pre_physics_callback=None
                    except:
                        pass
                if cls._sample._pre_physics_callback is None:
                    sample._pre_physics_callback=cls.pre_physics_callback
            super().on_follow_target_button_event(val,callback_fn=partial(_pre_physics_callback,val))
            return

        if cls._left_gripper_action_pressed(input_data) and not cls.already_pressed["Left Gripper Action"]:
            # if cls.already_pressed["Left Gripper Action"] is not True:
            #     pass
            cls.already_pressed["Left Gripper Action"]=True;
            val=cls.buttons["Left Gripper Action"].get_or_set_state()
            cls.on_gripper_action_button_event(name="Left", open=val)

        if cls._right_gripper_action_pressed(input_data) and not cls.already_pressed["Right Gripper Action"]:
            # if cls.already_pressed["Right Gripper Action"] is not True:
            #     pass
            cls.already_pressed["Right Gripper Action"]=True;
            val=cls.buttons["Right Gripper Action"].get_or_set_state()
            cls.on_gripper_action_button_event(name="Right", open=val)


    @classmethod
    def _reset_pressed(cls, input_data):
        # input_data format from metaqust3 class
        # # ~~the rot from metaquest3 class was {-rightRot.z, rightRot.x, -rightRot.y, rightRot.w}~~
        # {'left': {'pos': [1.0, 0.0, 0.0], 'rot': [0.0, 0.0, -1.0, 0.0], 'index_trigger': False, 'hand_trigger': False}, 
        # 'right': {'pos': [1.0, 0.0, 0.0], 'rot': [0.0, 0.0, -1.0, 0.0], 'index_trigger': False, 'hand_trigger': False}, 
        # 'A': False, 'B': False, 'X': False, 'Y': False}
        return not cls.old_input_data["X"] and input_data["X"]

    @classmethod
    def _follow_target_pressed(cls, input_data):
        return not cls.old_input_data["Y"] and input_data["Y"]
    
    @classmethod
    def _reposition_pressed(cls, input_data):
        # was A, but A is reserved in simpub/irxr unity
        return input_data["left"]["hand_trigger"] or input_data["right"]["hand_trigger"]
    
    @classmethod
    def _left_reposition_pressed(cls, input_data):
        # was A, but A is reserved in simpub/irxr unity
        return input_data["left"]["hand_trigger"]
    

    @classmethod
    def _right_reposition_pressed(cls, input_data):
        # was A, but A is reserved in simpub/irxr unity
        return input_data["right"]["hand_trigger"]
    
    @classmethod
    def _extras_logging(cls):
        input_data= cls._get_input_data()
        return {"left_reposition_pressed":cls._left_reposition_pressed(input_data),
                "right_reposition_pressed":cls._right_reposition_pressed(input_data)
                }
    @classmethod
    def _simulation_pressed(cls, input_data):
        return not cls.old_input_data["B"] and input_data["B"]

    @classmethod
    def _left_gripper_action_pressed(cls, input_data):
        return not cls.old_input_data["left"]["index_trigger"] and input_data["left"]["index_trigger"]

    @classmethod
    def _right_gripper_action_pressed(cls, input_data):
        return not cls.old_input_data["right"]["index_trigger"] and input_data["right"]["index_trigger"]

    @classmethod
    def _zeroing_pose(cls, name, input_data):
        cls.old_input_pos[name],cls.old_input_rot[name] = cls.get_pose(name, input_data)

    @classmethod
    def zeroing_pose(cls, input_data,name=None):
        if name is None:
            for _str in ["Left","Right"]:
                cls._zeroing_pose(_str,input_data)
        else:
            cls._zeroing_pose(name,input_data)

    @classmethod
    def transform_target(cls,input_data):
        #print(input_data is None)
        #async def _transform_target_async(cls,input_data):
        if input_data is None:
            return

        if cls.old_input_pos is None or cls.old_input_rot is None:
            cls.old_input_pos={}
            cls.old_input_rot={}
            cls.zeroing_pose(input_data)
            return
        
        # print(cls._reposition_pressed(input_data))
        # no pose update when holding the reposition button to relocate the followed target
        # if  cls._reposition_pressed(input_data):
        #     cls.zeroing_pose(input_data)
        #     return

        if  cls._left_reposition_pressed(input_data):
            cls.zeroing_pose(input_data, "Left")
        
        if  cls._right_reposition_pressed(input_data):
            cls.zeroing_pose(input_data, "Right")
        
        
        # print("--------")
        observations=cls._sample._world.get_observations()
        for _str in ["Left","Right"]:
            old_input_pos,old_input_rot=cls.old_input_pos[_str],cls.old_input_rot[_str]
            input_pos,input_rot=cls.old_input_pos[_str],cls.old_input_rot[_str] = cls.get_pose(_str,input_data)
            #print(f"input_pos is {input_pos}")

            # map the changes in vr controller to gripper pose via 4d quaternion
            delta_pos=np.subtract(input_pos,old_input_pos)

            # make translation intuitive: align the motion of the vr controller with motion of gripper
            delta_pos=np.array([-delta_pos[0],-delta_pos[1],delta_pos[2]])
            delta_rot=mu.mul(input_rot,mu.inverse(old_input_rot)) # ~~the order is unclear in doc could be another way around~~

            # make rotation intuitive, align with isaac sim gui
            # axis,angle=_q2aa(delta_rot)
            # delta_rot=_aa2q([axis[0],-axis[1],-axis[2]],angle)

            old_target_pos,old_target_rot=observations[cls._sample._target_name[_str]]["position"],observations[cls._sample._target_name[_str]]["orientation"]
            # target_pos=old_target_pos+delta_pos
            target_pos=np.add(old_target_pos,delta_pos)
            target_rot=mu.mul(delta_rot,old_target_rot)
            # set params expects a scalar-first (w, x, y, z) quaternion
            # the singe_prim_wrapper does not expect target pos/ori but delta!!! follow_target.py has to adapt
            # ~~lul, isaac core modules are not compatible~~
            # AsyncUtils._sample._task.set_params(target_position=target_pos,target_orientation=target_rot)
            # AsyncUtils._sample._task.set_params(delta_pos,delta_rot)

            # z axis should never below 0, better to use stage.get_stage_up_axis()-> str to determine up 
            # but z is the default up axis, set by set_stage_up_axis() in simulation context.py
            # dont follow it once it went below the ground
            eps= cls._sample._eps
            if target_pos[2]<=eps:
                target_pos[2]=eps
            if np.sum(np.abs(delta_pos)> np.finfo(np.dtype(delta_pos[0])).eps):
                # print(_str)
                # print(f"input pos: {input_pos}")
                # print(f"delta pos: {delta_pos}")
                # print(f"old tgt pos: {old_target_pos}")
                # print(f"tgt pos: {target_pos}")
                cls._sample._target[_str].set_local_pose(translation=target_pos, orientation=target_rot) 
            #await update_stage_async()
        #asyncio.ensure_future(_transform_target_async(cls,input_data))

    @classmethod
    def get_pose(cls,name,input_data):
        # rot is quaternion
        # ~~the rot from metaquest3 class was {-rightRot.z, rightRot.x, -rightRot.y, rightRot.w}~~
        # isaac sim expects {w, x, y, z} format
        # input_data format from metaqust3 class
        # ~~# the rot from metaquest3 class was {-rightRot.z, rightRot.x, -rightRot.y, rightRot.w}~~
        # {'left': {'pos': [1.0, 0.0, 0.0], 'rot': [0.0, 0.0, -1.0, 0.0], 'index_trigger': False, 'hand_trigger': False}, 
        # 'right': {'pos': [1.0, 0.0, 0.0], 'rot': [0.0, 0.0, -1.0, 0.0], 'index_trigger': False, 'hand_trigger': False}, 
        # 'A': False, 'B': False, 'X': False, 'Y': False}
        try:
            q_xyzw=input_data[name.lower()]["rot"]
            # correct the handedness
            # q_wxyz=np.array([q_xyzw[3],-q_xyzw[1],-q_xyzw[0],-q_xyzw[2]])
            # q_wxyz=np.array([q_xyzw[3],q_xyzw[1],q_xyzw[0],-q_xyzw[2]])
            q_wxyz=np.array([q_xyzw[3],-q_xyzw[0],q_xyzw[1],-q_xyzw[2]])

            p_xyz=input_data[name.lower()]["pos"]
        except Exception as e:
            print(e)
            print(">>> above error likely due to metaquest3 was called before simulation is started <<<")
            # metaquest3.py returns none in the data :( the err: TypeError: cannot unpack non-iterable NoneType object
            # only if when it is inited before simulation ever started
            return
        return p_xyz,q_wxyz

    @classmethod    
    def tearDown(cls):
        cls.unbind_inputs()
        try:
            cls.destroy_publisher()
        except:
            pass

    @classmethod
    def unbind_inputs(cls):
        try:
            _sample = cls._sample
            del _sample._pre_physics_callback
            _sample._pre_physics_callback=None
        except:
            pass



from omni.isaac.core.prims import XFormPrim
import carb
from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
from omni.physx.scripts import physicsUtils, utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.prims import RigidPrim
# from omni.isaac.core.articulations import Articulation
import itertools, random

# based on extsphysics\omni.physx.demos\omni\physxdemos\scenes\RigidBodyRopesDemo.py
class RigidBodyRope:
    def __init__(
        self,
        _world,
        _scene_name="RopeScene",
        _rope_name="Rope",
        _linkHalfLength=0.018, 
        _linkRadius=None, #0.013,
        _ropeLength=0.8,
        _rope_damping=0.6, # lower the better, otherwise: bump in the rope
        _rope_stiffness=0.1, # lower: rope like, higher: rigid line
        _rope_y_pos=None,
        _coneAngleLimit=140,
        _ropeColor=None,
        _density=None, # 0.000000000000005,
        _randomize=False,
        _randomize_on_reset=False,
    ):
        self._world = _world
        self._stage=_world.stage
        self._scene= _world.scene
        self._scene_name=_scene_name
        self._rope_name=_rope_name
        self._scene_path=f"/World/{_scene_name}" #stage_utils.get_next_free_path(f"/World/{name}")
        self._linkHalfLength = _linkHalfLength
        self._linkRadius = _linkRadius or 0.75 * self._linkHalfLength
        self._ropeLength = _ropeLength
        self._rope_damping = _rope_damping
        self._rope_stiffness = _rope_stiffness
        self._rope_y_pos=_rope_y_pos
        self._coneAngleLimit = _coneAngleLimit
        self._ropeColor =  _ropeColor or Gf.Vec3f(165.0, 21.0, 21.0)/255.0
        self._density = _density
        self._randomize=_randomize
        self._randomize_on_reset=_randomize_on_reset
        self._capsules=[]
        self._default_state={}
        UsdGeom.Scope.Define(_world.stage, self._scene_path)

        # physics options:
        self._contactOffset = 0.01
        self._physicsMaterialPath = Sdf.Path(self._scene_path).AppendChild("RopePhysicsMaterial")
        UsdShade.Material.Define(self._stage, self._physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._physicsMaterialPath))
        material.CreateStaticFrictionAttr().Set(0.4) # 1.0
        material.CreateDynamicFrictionAttr().Set(0.1) # 0.1
        material.CreateRestitutionAttr().Set(0.001) # just be alittle bouncy

        _world._rope=self

    def deleteRope(self):
        if is_prim_path_valid(self._rope_path):
            prims_utils.delete_prim(self._rope_path)
        del self._capsules
        del self._default_state
        self._capsules=[]
        self._default_state={}

    def get_world_pose(self):
        world_positions=[]
        world_orientations=[]
        for rigidPrim in self._capsules:
            # get_world_pose returns a tuple of np.array
            # tolist(): np.array not json serializable
            _world_pose=rigidPrim.get_world_pose()
            world_positions.append(_world_pose[0].tolist())
            world_orientations.append(_world_pose[1].tolist())
        return [world_positions,world_orientations]
    
    def get_local_pose(self):
        local_positions=[]
        local_orientations=[]
        for rigidPrim in self._capsules:
            # get_world_pose returns a tuple of np.array
            # tolist(): np.array not json serializable
            _local_pose=rigidPrim.get_local_pose()
            local_positions.append(_local_pose[0].tolist())
            local_orientations.append(_local_pose[1].tolist())
        return [local_positions,local_orientations]

    def get_local_position(self):
        local_positions=[]
        for rigidPrim in self._capsules:
            # get_world_pose returns a tuple of np.array
            # tolist(): np.array not json serializable
            _local_pose=rigidPrim.get_local_pose()
            local_positions.append(_local_pose[0].tolist())
        return local_positions
    
    def get_world_position(self):
        world_positions=[]
        for rigidPrim in self._capsules:
            # get_world_pose returns a tuple of np.array
            # tolist(): np.array not json serializable
            _world_pose=rigidPrim.get_world_pose()
            world_positions.append(_world_pose[0].tolist())
        return world_positions
    
    def get_world_orientation(self):
        world_orientations=[]
        for rigidPrim in self._capsules:
            # get_world_pose returns a tuple of np.array
            # tolist(): np.array not json serializable
            _world_pose=rigidPrim.get_world_pose()
            world_orientations.append(_world_pose[1].tolist())
        return world_orientations

    def set_world_pose(self,positions=None,orientations=None):
        for i,rigidPrim in enumerate(self._capsules):
            position = None if positions is None else positions[i]
            orientation = None if orientations is None else orientations[i]
            rigidPrim.set_world_pose(position=position,orientation=orientation)

    def _to_randomize(self,x=None,y=None,z=None):
        x=x or self.x
        y=y or self.y
        z=z or self.z
        nums = list(itertools.chain(
            range(-2,0),
            range(0,2)
            ))
        pos_offset=random.choices(nums, k=3)
        nums = list(itertools.chain(
            range(-180,-90),
            range(90,180)
            ))
        angles=random.choices(nums, k=3)
        position=Gf.Vec3d(x+pos_offset[0]/10, y+pos_offset[1]/10, z+pos_offset[2]/10)
        orientation=Gf.Vec3d(angles[0],angles[1],angles[2])
        return position, orientation
    
    def post_reset(self):
        # TODO to investigate why the default post_reset() does not reset to the default state
        for linkInd, rigidPrim in enumerate(self._capsules):
            if self._randomize_on_reset:
                linkInd=len(self._capsules)//2
                linkLength=self._linkLength
                xStart=self._xStart
                x = xStart + linkInd * linkLength
                position,orientation=self._to_randomize(x)
                self._capsules[linkInd].set_default_state(position=list(position),orientation=euler_angles_to_quat(list(orientation)))
            
            default=rigidPrim.get_default_state()
            position=default.position
            orientation=default.orientation
            rigidPrim.set_world_pose(position=position,orientation=orientation)
            # isaac-sim-4.2.0/exts/omni.isaac.core/omni/isaac/core/utils/numpy/tensor.py", line 60, in tensor_cat
            #     return np.concatenate(data, axis=dim)
            # ValueError: zero-dimensional arrays cannot be concatenated
            # rigidPrim.post_reset()

    def createRope(self):
        self._linkLength = linkLength = 2.0 * self._linkHalfLength - self._linkRadius
        numLinks = int(self._ropeLength / linkLength)
        self._xStart = xStart = -numLinks * linkLength * 0.5

        _rope_name=self._rope_name
        self._rope_path=xformPath=stage_utils.get_next_free_path(_rope_name, self._scene_path)
        UsdGeom.Xform.Define(self._stage, xformPath)
        # self._rope=XFormPrim(xformPath)
        # # register to the scene # TODO as articulation subclass??
        # self._scene.add(self._rope)
        self.y=y =self._rope_y_pos or -0.05 #0.7
        self.z=z = 0.5
        self.x=0.0

        # Create individual capsules 
        # current impl of phy fabric does not support pointinstancer
        capsules = []
        self._capsules=[]
        for linkInd in range(numLinks):
            x = xStart + linkInd * linkLength
            capsulePath = Sdf.Path(xformPath).AppendChild(f"capsule_{linkInd}")
            capsule = self._createCapsule(capsulePath)
            # Set transform for each capsule
            xformable = UsdGeom.Xformable(capsule)
            if self._randomize:
                position,orientation=self._to_randomize(x,y,z)   
                # xformable.AddTranslateOp().Set(Gf.Vec3d(x+random.uniform(-0.2, 0.2), y, z+random.uniform(-0.2, 0.2)))
                # xformable.AddRotateXYZOp().Set(Gf.Vec3d(random.uniform(-180.0, 180.0),random.uniform(-180.0, 180.0),random.uniform(-180.0, 180.0)))
            else:
                position=Gf.Vec3d(x, y, z)
                orientation=Gf.Vec3d(0,0,0)
            xformable.AddTranslateOp().Set(position)
            # TODO >>> omni.hydra.scene_delegate.plugin] cannot find xform op xformOp:rotateXYZ for /World/RopeScene/Rope/capsule_
            # this warning can be safely ignored, the orientation is set despite this warning
            rotation_op=xformable.AddRotateXYZOp().Set(orientation)
            capsules.append(capsulePath)
            _capsule_prim=RigidPrim(capsulePath.pathString)
            self._capsules.append(_capsule_prim)
            _capsule_prim.set_default_state(position=list(position),orientation=euler_angles_to_quat(list(orientation)))

        # Create joints between capsules
        jointX = self._linkHalfLength - 0.5 * self._linkRadius
        for linkInd in range(numLinks - 1):
            jointPath = Sdf.Path(xformPath).AppendChild(f"joint_{linkInd}")
            joint = self._createJoint(jointPath)
            
            # Set joint properties
            joint.CreateBody0Rel().SetTargets([capsules[linkInd]])
            joint.CreateBody1Rel().SetTargets([capsules[linkInd + 1]])
            
            # Set local positions for joint ends
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(jointX, 0, 0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-jointX, 0, 0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
            # PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateJointFrictionAttr().Set(0.001) # NEW! #joint friction only works in articulation


    def _createCapsule(self, path: Sdf.Path):
        capsuleGeom = UsdGeom.Capsule.Define(self._stage, path)
        capsuleGeom.CreateHeightAttr(self._linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self._linkRadius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.CreateDisplayColorAttr().Set([self._ropeColor])

        # Add physics properties
        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        # massAPI.CreateDensityAttr().Set(self._density) # TODO BUG ?
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self._contactOffset)
        physicsUtils.add_physics_material_to_prim(self._stage, capsuleGeom.GetPrim(), self._physicsMaterialPath)
        
        return capsuleGeom

    def _createJoint(self, jointPath):
        joint = UsdPhysics.Joint.Define(self._stage, jointPath)
        d6Prim = joint.GetPrim()

        # Lock translational DOFs
        for axis in ["transX", "transY", "transZ"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, axis)
            limitAPI.CreateLowAttr(1.0)
            limitAPI.CreateHighAttr(-1.0)

        # Configure rotational DOFs
        for axis in ["rotX", "rotY", "rotZ"]:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, axis)
            limitAPI.CreateLowAttr(-self._coneAngleLimit)
            limitAPI.CreateHighAttr(self._coneAngleLimit)

            # Add joint drives
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, axis)
            driveAPI.CreateTypeAttr("acceleration") # better than force
            driveAPI.CreateDampingAttr(self._rope_damping)
            driveAPI.CreateStiffnessAttr(self._rope_stiffness)

        return joint
    
# fabric does not support pointinstancer: thus discarded
class RigidBodyRope_withpointinstancer:
    def __init__(
        self,
        _world,
        _scene_name="RopeScene",
        _rope_name="Rope",
        _linkHalfLength=0.025,
        _linkRadius=None,
        _ropeLength=2.0,
        _rope_damping=10,
        _rope_stiffness=1.0,
        _coneAngleLimit=100,
        _ropeColor=None,
        _density=0.00005,
    ):
        self._world = _world
        self._stage=_world.stage
        self._scene= _world.scene
        self._scene_name=_scene_name
        self._rope_name=_rope_name
        self._scene_path=f"/World/{_scene_name}" #stage_utils.get_next_free_path(f"/World/{name}")
        self._linkHalfLength = _linkHalfLength
        self._linkRadius = _linkRadius or 0.5 * self._linkHalfLength
        self._ropeLength = _ropeLength
        self._rope_damping = _rope_damping
        self._rope_stiffness = _rope_stiffness
        self._coneAngleLimit = _coneAngleLimit
        self._ropeColor =  _ropeColor or Gf.Vec3f(165.0, 21.0, 21.0)/255.0
        self._density = _density
        UsdGeom.Scope.Define(_world.stage, self._scene_path)

        # physics options:
        self._contactOffset = 0.02
        self._physicsMaterialPath = Sdf.Path(self._scene_path).AppendChild("RopePhysicsMaterial")
        UsdShade.Material.Define(self._stage, self._physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(self._stage.GetPrimAtPath(self._physicsMaterialPath))
        material.CreateStaticFrictionAttr().Set(0.4)
        material.CreateDynamicFrictionAttr().Set(0.1)
        material.CreateRestitutionAttr().Set(0)
    
    def deleteRope(self):
        prims_utils.delete_prim(self._scene_name)

    def get_world_pose(self):
        pass
    
    def get_local_pose(self):
        pass


    def get_local_position(self):
        pass
        # from pxr import Usd, UsdGeom
        # import omni.usd
        # from omni.isaac.core.utils.prims import get_prim_at_path
        # instancer_path = "/World/RopeScene/Rope/RigidBodyInstancer"

        # # Get the joint prim
        # instancer_prim = get_prim_at_path(instancerPath)
        # point_instancer = UsdGeom.PointInstancer(instancer_prim)
        # rel=point_instancer.GetPositionsAttr().Get()
        # print(rel)
    
    def get_world_position(self):
        pass
    
    def createRope(self):    
        linkLength = 2.0 * self._linkHalfLength - self._linkRadius
        numLinks = int(self._ropeLength / linkLength)
        xStart = -numLinks * linkLength * 0.5

        _rope_name=self._rope_name
        self._rope_path=xformPath=stage_utils.get_next_free_path(_rope_name, self._scene_path)
        UsdGeom.Xform.Define(self._stage, xformPath)

        # capsule instancer
        self._rboinstancer_path=instancerPath = Sdf.Path(xformPath).AppendChild("RigidBodyInstancer")
        # api: kit\dev\fabric\include\usdrt\scenegraph\usd\usdGeom\pointInstancer.h
        rboInstancer = UsdGeom.PointInstancer.Define(self._stage, instancerPath)

        capsulePath = instancerPath.AppendChild("Capsule")
        self._createCapsule(capsulePath)

        meshIndices = []
        positions = []
        orientations = []

        y = 0.0 
        z = 0.5

        capsules = []
        self._capsules=[]
        for linkInd in range(numLinks):
            meshIndices.append(0)
            x = xStart + linkInd * linkLength
            positions.append(Gf.Vec3f(x, y, z))
            orientations.append(Gf.Quath(1.0))

        meshList = rboInstancer.GetPrototypesRel()
        # add mesh reference to point instancer
        meshList.AddTarget(capsulePath)

        rboInstancer.GetProtoIndicesAttr().Set(meshIndices)
        rboInstancer.GetPositionsAttr().Set(positions)
        rboInstancer.GetOrientationsAttr().Set(orientations)

        # joint instancer
        self._jointinstancer_path=jointInstancerPath = Sdf.Path(xformPath).AppendChild("JointInstancer")
        jointInstancer = PhysxSchema.PhysxPhysicsJointInstancer.Define(self._stage, jointInstancerPath)

        jointPath = jointInstancerPath.AppendChild("Joint")
        self._createJoint(jointPath)

        meshIndices = []
        body0s = []
        body0indices = []
        localPos0 = []
        localRot0 = []
        body1s = []
        body1indices = []
        localPos1 = []
        localRot1 = []      
        body0s.append(instancerPath)
        body1s.append(instancerPath)

        jointX = self._linkHalfLength - 0.5 * self._linkRadius
        for linkInd in range(numLinks - 1):
            meshIndices.append(0)

            body0indices.append(linkInd)
            body1indices.append(linkInd + 1)
 
            localPos0.append(Gf.Vec3f(jointX, 0, 0)) 
            localPos1.append(Gf.Vec3f(-jointX, 0, 0)) 
            localRot0.append(Gf.Quath(1.0))
            localRot1.append(Gf.Quath(1.0))

        meshList = jointInstancer.GetPhysicsPrototypesRel()
        meshList.AddTarget(jointPath)
        # api: kit\dev\fabric\include\usdrt\scenegraph\usd\physxSchema\physxPhysicsJointInstancer.h
        jointInstancer.GetPhysicsProtoIndicesAttr().Set(meshIndices)

        jointInstancer.GetPhysicsBody0sRel().SetTargets(body0s)
        jointInstancer.GetPhysicsBody0IndicesAttr().Set(body0indices)
        jointInstancer.GetPhysicsLocalPos0sAttr().Set(localPos0)
        jointInstancer.GetPhysicsLocalRot0sAttr().Set(localRot0)

        jointInstancer.GetPhysicsBody1sRel().SetTargets(body1s)
        jointInstancer.GetPhysicsBody1IndicesAttr().Set(body1indices)
        jointInstancer.GetPhysicsLocalPos1sAttr().Set(localPos1)
        jointInstancer.GetPhysicsLocalRot1sAttr().Set(localRot1)

    
    def _createCapsule(self, path: Sdf.Path):
        capsuleGeom = UsdGeom.Capsule.Define(self._stage, path)
        capsuleGeom.CreateHeightAttr(self._linkHalfLength)
        capsuleGeom.CreateRadiusAttr(self._linkRadius)
        capsuleGeom.CreateAxisAttr("X")
        capsuleGeom.CreateDisplayColorAttr().Set([self._ropeColor])

        UsdPhysics.CollisionAPI.Apply(capsuleGeom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(capsuleGeom.GetPrim())
        massAPI = UsdPhysics.MassAPI.Apply(capsuleGeom.GetPrim())
        massAPI.CreateDensityAttr().Set(self._density)
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(capsuleGeom.GetPrim())
        physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
        physxCollisionAPI.CreateContactOffsetAttr().Set(self._contactOffset)
        physicsUtils.add_physics_material_to_prim(self._stage, capsuleGeom.GetPrim(), self._physicsMaterialPath)

    def _createJoint(self, jointPath):        
        joint = UsdPhysics.Joint.Define(self._stage, jointPath)

        # locked DOF (lock - low is greater than high)
        d6Prim = joint.GetPrim()
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
        limitAPI.CreateLowAttr(1.0)
        limitAPI.CreateHighAttr(-1.0)
        # limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
        # limitAPI.CreateLowAttr(1.0)
        # limitAPI.CreateHighAttr(-1.0)

        # Moving DOF:
        dofs = ["rotX", "rotY", "rotZ"]
        for d in dofs:
            limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
            limitAPI.CreateLowAttr(-self._coneAngleLimit)
            limitAPI.CreateHighAttr(self._coneAngleLimit)

            # joint drives for rope dynamics:
            driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
            driveAPI.CreateTypeAttr("force")
            driveAPI.CreateDampingAttr(self._rope_damping)
            driveAPI.CreateStiffnessAttr(self._rope_stiffness)

# TODO
class RigidBodyRope_Factory():
    pass


# TODO deformable have to work on GPU, which means no numpy code!
# from omni.kit.commands import execute
# class DeformableRope:
#     def __init__(self,_stage):
#         self._stage=_stage
#         pass

#     def createRope(self):
#         stage = self._stage

#         # Define the root Xform
#         rootXform = UsdGeom.Xform.Define(stage, '/World')

#         cylinderXform = UsdGeom.Xform.Define(stage, '/World/DeformableRope')
#         cylinderXform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
#         cylinderXform.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
#         cylinderXform.AddScaleOp().Set(Gf.Vec3f(0.03, 5, 0.03))


#         result, prim_path = execute(
#             "CreateMeshPrim",
#             prim_type="Cylinder",
#             object_origin=Gf.Vec3f(0.0, 0.0, 0.0),
#         )
#         execute("MovePrim", path_from=prim_path, path_to='/World/DeformableRope/DeformableRope')

#         cylinderPrim = stage.GetPrimAtPath("/World/DeformableRope/DeformableRope")

#         PhysxSchema.PhysxDeformableBodyAPI.Apply(cylinderPrim)
#         cylinderPrim.CreateAttribute('physxDeformable:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
#         cylinderPrim.CreateAttribute('physxDeformable:numberOfTetsPerHex', Sdf.ValueTypeNames.UInt).Set(5)
#         cylinderPrim.CreateAttribute('physxDeformable:simulationHexahedralResolution', Sdf.ValueTypeNames.UInt).Set(50)


#         # Define Material for deformable physics using PhysxSchema
#         # cotton
#         # https://repository.gatech.edu/server/api/core/bitstreams/8214a46f-0424-436e-a261-5c46f91c8d4e/content
#         # http://www-materials.eng.cam.ac.uk/mpsite/short/OCR/ropes/default.html
#         material = stage.DefinePrim('/deformablePhysicsMaterial', 'Material')
#         PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(material)
#         material.CreateAttribute('physxDeformableBodyMaterial:density', Sdf.ValueTypeNames.Float).Set(1.54)
#         material.CreateAttribute('physxDeformableBodyMaterial:dynamicFriction', Sdf.ValueTypeNames.Float).Set(0.257)
#         material.CreateAttribute('physxDeformableBodyMaterial:elasticityDamping', Sdf.ValueTypeNames.Float).Set(0.3)
#         material.CreateAttribute('physxDeformableBodyMaterial:poissonsRatio', Sdf.ValueTypeNames.Float).Set(0.0)
#         material.CreateAttribute('physxDeformableBodyMaterial:youngsModulus', Sdf.ValueTypeNames.Float).Set(8000000000)
#         pass





# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka

# based on exts\omni.isaac.franka\omni\isaac\franka\tasks\follow_target.py
class FollowTarget(tasks.FollowTarget):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "franka_follow_target".
        target_prim_path (Optional[str], optional): [description]. Defaults to None.
        target_name (Optional[str], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        franka_prim_path (Optional[str], optional): [description]. Defaults to None.
        franka_robot_name (Optional[str], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str = "franka_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        franka_prim_path: Optional[str] = None,
        franka_robot_name: Optional[str] = None,
        # new added params for setting up robot
        franka_position: Optional[np.ndarray] = None,
        franka_orientation: Optional[np.ndarray] = None,
        franka_gripper_open_position: Optional[np.ndarray] = None,
        franka_gripper_closed_position: Optional[np.ndarray] = None,
        franka_deltas: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        self._franka_prim_path = franka_prim_path
        self._franka_robot_name = franka_robot_name

        self._franka_position=franka_position
        self._franka_orientation=franka_orientation
        self._franka_gripper_open_position=franka_gripper_open_position
        self._franka_gripper_closed_position=franka_gripper_closed_position
        self._franka_deltas=franka_deltas

        return

    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        if self._franka_prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
        if self._franka_robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
        franka=Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name, 
        position=self._franka_position,
        orientation=self._franka_orientation,
        gripper_open_position=self._franka_gripper_open_position,
        gripper_closed_position=self._franka_gripper_closed_position,
        deltas=self._franka_deltas,)
        # see extscache\omni.importer.urdf-XXX.cp310\omni\importer\urdf\scripts\samples\import_franka.py
        #   PhysxSchema.PhysxArticulationAPI.Get(stage, "/panda").CreateSolverPositionIterationCountAttr(64)
        #   PhysxSchema.PhysxArticulationAPI.Get(stage, "/panda").CreateSolverVelocityIterationCountAttr(64)
        # the urdf importer set it to 64, causing 
        #   1) performance issue, that is 64 times too many calculations
        #   2) franka explosion: only the base/root remains in the scene 
        #       due to velocity accumulation if the target gripper position is unreachable, like inside a rigidbody/below groundplane
        franka.set_solver_position_iteration_count(4)
        franka.set_solver_velocity_iteration_count(0)
        return franka
