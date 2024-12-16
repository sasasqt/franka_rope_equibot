import json
import numpy as np
import os
from math import sqrt
import math

input_dir = r"C:\Users\Shadow\project\franka_rope\demos\02_train"
output_dir = r"C:\Users\Shadow\project\franka_data\abspos_train_straight_right"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
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


gravity_dir = [0, 0, -1]  # z is up in isaac sim


def normalize_quat(q):
    norm = sqrt(
        float(q[0]) ** 2 + float(q[1]) ** 2 + float(q[2]) ** 2 + float(q[3]) ** 2
    )
    q[0], q[1], q[2], q[3] = q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm
    return q


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


def q2cols(q):
    q = normalize_quat(q)
    w, x, y, z = q
    col1 = [2 * (w**2 + x**2) - 1, 2 * (x * y + w * z), 2 * (x * z - w * y)]
    col3 = [2 * (w * y + x * z), 2 * (y * z - w * x), w**2 - x**2 - y**2 + z**2]
    return col1, col3


def quat_conj(q):
    q[1], q[2], q[3] = -q[1], -q[2], -q[3]
    return q


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [w, x, y, z]


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
        rope_pos = np.array(curr["Rope"]["Rope_world_position"])  # as pc
        rope_rot = np.array(
            curr["Rope"]["Rope_world_orientation"]
        )  # cannot be integrated in?

        if (
            curr["Right"]["applied_joint_positions"][-1] < 0.025
        ):  # 0/-0.3 is closed, ~0.05 is opened
            gripper_action = 0
        else:
            gripper_action = 1

        # should be like (3460, 3)
        pc = np.array(rope_pos)
        # recording["pc"].append(pc)
        delta_pos = (
            np.array(fut["Right"]["Right_target_world_position"])
            - right_target_world_pos
        )
        delta_rot = np.array(
            quat_mul(
                normalize_quat(fut["Right"]["Right_target_world_orientation"]),
                quat_conj(normalize_quat(right_target_world_rot)),
            )
        )

        # should be like (2, 7)
        action = np.array(
            [
                gripper_pose,
                delta_pos[0],
                delta_pos[1],
                delta_pos[2],
                quat2rpy(delta_rot)[0],
                quat2rpy(delta_rot)[1],
                quat2rpy(delta_rot)[2],
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
