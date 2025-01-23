import json
import numpy as np
import os
from math import sqrt
import math
import hydra
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.utils._isaac_utils import math as mu


# len(data) = 1
# data[0].keys()=Isaac Sim Data
# len(data[0]['Isaac Sim Data']) = saved steps
# data[0]['Isaac Sim Data'] = list
# data[0]['Isaac Sim Data'][i].keys() = ['futent_time', 'futent_time_step', 'data']
# data[0]['Isaac Sim Data'][i]['data'].keys()=dict_keys(['Left', 'Right', 'Rope', 'extras', 'Datetime'])
# data[0]['Isaac Sim Data'][i]['data']['Left']=dict_keys(['Left_joint_positions', 'applied_joint_positions', 'Left_end_effector_world_position', 
#   'Left_end_effector_world_orientation', 'Left_end_effector_local_position', 'Left_end_effector_local_orientation', 'Left_target_world_position', 
#   'Left_target_world_orientation', 'Left_target_local_position', 'Left_target_local_orientation'])
# data[0]['Isaac Sim Data'][i]['data']['Rope']=dict_keys(['Rope_world_position', 'Rope_world_orientation'])
# data[0]['Isaac Sim Data'][i]['data']['extras']=dict_keys(['left_reposition_pressed', 'right_reposition_pressed'])
# recording={}
# recording["pc"]=[]
# recording["action"]=[]
# recording["eef_pos"]=[]


def _l2_norm(q):
    return sqrt(sum(map(lambda x: float(x)**2, q)))

def normalize_quat(q):
    norm = _l2_norm(q)
    q[0], q[1], q[2], q[3] = q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm
    return q

def q2aa(q):
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

def quat2rpy(q):
    q = normalize_quat(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    # roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

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

def q2cols(q):
    q = normalize_quat(q)
    w, x, y, z = q
    col1 = [2 * (w**2 + x**2) - 1, 2 * (x * y + w * z), 2 * (x * z - w * y)]
    col3 = [2 * (w * y + x * z), 2 * (y * z - w * x), w**2 - x**2 - y**2 + z**2]
    return col1, col3


def quat_conj(q):
    q[1], q[2], q[3] = -q[1], -q[2], -q[3]
    return q

# this somehow does not match the result computed using mu.mul BUG?
# mu.mul is used to calculate obs
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [w, x, y, z]

@hydra.main(config_path="equibot/policies/configs", config_name="franka_base")
def main(cfg):
    input_dir = cfg.franka_rope.preprocess.input_dir
    output_dir = cfg.franka_rope.preprocess.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    rel=eval(str(cfg.franka_rope.preprocess.rel).title())
    rpy=eval(str(cfg.franka_rope.preprocess.rpy).title())

    gravity_dir = [0, 0, -1]  # z is up in isaac sim

    for ep, filename in enumerate(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue

        file = os.path.join(input_dir, filename)

        data = []
        with open(file, "r") as f:
            for line in f:
                data.append(json.loads(line))

        # to mimic saved npz with keys pc, rgb?, action, eef_pos
        for i, _fut in enumerate(data[0]["Isaac Sim Data"]):
            fut = _fut["data"]
            if i == 0:
                curr = fut
                if (
                    curr["Right"]["applied_joint_positions"][-1] < 0.025
                ):  # 0/-0.3 is closed, ~0.5 is opened
                    gripper_action = 0
                    gripper_pose = 0
                else:
                    gripper_action = 1
                    gripper_pose = 1

                continue

            gripper_pose = gripper_action
            franka_joints = np.array(
                curr["Right"]["Right_joint_positions"]
            )  # not exposed to the algorithm
            right_target_world_pos = np.array(
                curr["Right"]["Right_target_world_position"]
            )  # as-is
            right_target_world_rot = np.array(
                curr["Right"]["Right_target_world_orientation"]
            )  # need to be convert to rotation matrix with 3rd col pointing downwards
            t_pc = np.array(curr["T"]["pc"])  # as pc

            if (
                curr["Right"]["applied_joint_positions"][-1] < 0.025
            ):  # 0/-0.3 is closed, ~0.05 is opened
                gripper_action = 0
            else:
                gripper_action = 1

            # should be like (3460, 3)
            pc = np.array(t_pc)
            if rel:
                # recording["pc"].append(pc)
                delta_pos = (
                    np.array(fut["Right"]["Right_target_world_position"])
                    - right_target_world_pos
                )
                # BAD IDEA
                # # cap fast movement 1) from human demo, 2) right after stationary
                # if (_l2_norm(delta_pos)*30.0>=0.03): 
                #     # print(delta_pos,_l2_norm(delta_pos),"???")
                #     delta_pos=delta_pos*0.03/30.0/_l2_norm(delta_pos)
                #     # print(delta_pos,_l2_norm(delta_pos),"???")

                delta_rot = np.array(
                    mu.mul(
                        ((fut["Right"]["Right_target_world_orientation"])),
                        mu.inverse((right_target_world_rot)), # quat_conj should close to mu.inverse(normalize_quat(right_target_world_rot))
                    )
                )
                if rpy:
                    ori=list(quat2rpy(delta_rot)) # [0.00042632472221359693, -0.002514371491837601, -3.139069198502559]
                    # handle discontinuity induced by mu.mul
                    if ori[2]<-2:
                        ori[2]+=np.pi
                        ori[2]*=-1
                        
                    if ori[2]>2:
                        ori[2]-=np.pi
                        ori[2]*=-1
                    print(ori)
                    print()

                    action = np.array(
                        [
                            gripper_pose,
                            delta_pos[0],
                            delta_pos[1],
                            delta_pos[2],
                            ori[0], # rpy2quat(quat2rpy(delta_rot)) should be close to delta_rot
                            ori[1],
                            ori[2],
                        ]
                    )
                else: 
                    reconstructed=mu.mul(delta_rot,right_target_world_rot) # should close to fut["Right"]["Right_target_world_orientation"]                
                    axis,angle=q2aa(delta_rot) # aa2q(axis,angle) should close to delta_rot, angle ~ 3.1x
                    angle/=np.pi
                    axis/=_l2_norm(axis)
                    aa=axis*angle
                    # t_angle=_l2_norm(aa)
                    # t_axis=aa/t_angle
                    # t_ori=mu.mul((np.array(aa2q(t_axis,t_angle))),(right_target_world_rot)) # should close to fut["Right"]["Right_target_world_orientation"]

                    # should be like (2, 7)
                    action = np.array(
                        [
                            gripper_pose,
                            delta_pos[0],
                            delta_pos[1],
                            delta_pos[2],
                            aa[0],
                            aa[1],
                            aa[2],
                        ]
                    )
            if not rel:
                # should be like (2, 7)
                abs_pos=fut["Right"]["Right_target_world_position"]
                abs_rot=fut["Right"]["Right_target_world_orientation"]

                if rpy:
                    action = np.array(
                        [
                            gripper_pose,
                            abs_pos[0],
                            abs_pos[1],
                            abs_pos[2],
                            abs_rot[0], # rpy2quat(quat2rpy(abs_rot)) should be close to abs_rot
                            abs_rot[1],
                            abs_rot[2],
                        ]
                    )
                else:         
                    axis,angle=q2aa(abs_rot) # angle ~3.1x
                    angle/=np.pi
                    axis/=_l2_norm(axis)
                    aa=axis*angle
                    action = np.array(
                        [
                            gripper_pose,
                            abs_pos[0],
                            abs_pos[1],
                            abs_pos[2],
                            aa[0],
                            aa[1],
                            aa[2],
                        ]
                    )

            # recording["action"].append(action)
            #  state/eef_pos needs to be like [eef_pos, dir1, dir2, gravity_dir, gripper_pose]
            # dir1, dir2: first and third column of end effector’s rotation matrix (orientation)
            col1, col3 = q2cols(right_target_world_rot)
            # should be like (2, 13)
            eef_pos = np.array(
                (
                    right_target_world_pos[0],
                    right_target_world_pos[1],
                    right_target_world_pos[2],
                    col1[0],
                    col1[1],
                    col1[2],
                    col3[0],
                    col3[1],
                    col3[2],
                    gravity_dir[0],
                    gravity_dir[1],
                    gravity_dir[2],
                    gripper_pose,
                )
            )
            # recording["eef_pos"].append(eef_pos)

            gripper_pose = gripper_action
            curr = fut

            np.savez(
                # :02d is expected from the dataset py
                os.path.join(output_dir + rf"\01_ep{ep:06d}_view0_t{i:02d}.npz"),
                pc=np.array(pc),
                action=np.array(action[np.newaxis, :]),
                eef_pos=np.array(eef_pos[np.newaxis, :]),
            )





if __name__ == "__main__":
    main()
