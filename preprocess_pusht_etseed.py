import json
import numpy as np
import os
from math import sqrt
import math
import hydra
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

# from omni.isaac.utils._isaac_utils import math as mu
import kornia

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
    return sqrt(sum(map(lambda x: float(x) ** 2, q)))


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
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

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


def q2rmat(q):
    q = normalize_quat(q)
    w, x, y, z = q

    w2 = w * w
    x2 = x * x
    y2 = y * y
    z2 = z * z

    r11 = w2 + x2 - y2 - z2
    r12 = 2 * (x * y - w * z)
    r13 = 2 * (x * z + w * y)

    r21 = 2 * (x * y + w * z)
    r22 = w2 - x2 + y2 - z2
    r23 = 2 * (y * z - w * x)

    r31 = 2 * (x * z - w * y)
    r32 = 2 * (y * z + w * x)
    r33 = w2 - x2 - y2 + z2

    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def rmat2q(rmat):
    trace = np.trace(rmat)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (rmat[2, 1] - rmat[1, 2]) / S
        y = (rmat[0, 2] - rmat[2, 0]) / S
        z = (rmat[1, 0] - rmat[0, 1]) / S
    elif (rmat[0, 0] > rmat[1, 1]) and (rmat[0, 0] > rmat[2, 2]):
        S = np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2]) * 2  # S = 4 * x
        w = (rmat[2, 1] - rmat[1, 2]) / S
        x = 0.25 * S
        y = (rmat[0, 1] + rmat[1, 0]) / S
        z = (rmat[0, 2] + rmat[2, 0]) / S
    elif rmat[1, 1] > rmat[2, 2]:
        S = np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2]) * 2  # S = 4 * y
        w = (rmat[0, 2] - rmat[2, 0]) / S
        x = (rmat[0, 1] + rmat[1, 0]) / S
        y = 0.25 * S
        z = (rmat[1, 2] + rmat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1]) * 2  # S = 4 * z
        w = (rmat[1, 0] - rmat[0, 1]) / S
        x = (rmat[0, 2] + rmat[2, 0]) / S
        y = (rmat[1, 2] + rmat[2, 1]) / S
        z = 0.25 * S

    return [w, x, y, z]


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

    # NEW
    tgt_pc = np.array(
        [
            [0.1515958048403263, 0.04015194624662399, 0.050000112503767014],
            [0.151595801115036, 0.04015194624662399, 1.1362135410308838e-07],
            [0.1765730157494545, 0.08346636593341827, 0.05000011436641216],
            [0.1765730157494545, 0.08346636407077312, 1.1175870895385742e-07],
            [0.11910998821258545, 0.05888485535979271, 0.050000112503767014],
            [0.11910998821258545, 0.05888485163450241, 1.1362135410308838e-07],
            [0.14408720284700394, 0.10219927318394184, 0.05000011436641216],
            [0.14408719912171364, 0.10219927225261927, 1.1175870895385742e-07],
            [0.08662417437881231, 0.07761775888502598, 0.050000112503767014],
            [0.08662417344748974, 0.07761775888502598, 1.0989606380462646e-07],
            [0.11160138435661793, 0.12093218229711056, 0.05000011436641216],
            [0.11160138342529535, 0.12093218229711056, 1.1175870895385742e-07],
            [0.05413835868239403, 0.09635066892951727, 0.050000112503767014],
            [0.05413835495710373, 0.0963506679981947, 1.0989606380462646e-07],
            [0.07911556959152222, 0.13966508954763412, 0.050000110641121864],
            [0.07911556959152222, 0.13966508582234383, 1.1175870895385742e-07],
            [0.021652542054653168, 0.11508357711136341, 0.050000112503767014],
            [0.021652542054653168, 0.11508357524871826, 1.0989606380462646e-07],
            [0.046629756689071655, 0.15839799493551254, 0.050000110641121864],
            [0.04662975296378136, 0.15839799493551254, 1.1175870895385742e-07],
            [0.20163410902023315, 0.12688086926937103, 0.050126830115914345],
            [0.20163410902023315, 0.12688086926937103, 0.0001268293708562851],
            [0.24494433030486107, 0.10189651697874069, 0.050126830115914345],
            [0.24494433030486107, 0.10189651697874069, 0.0001268293708562851],
            [0.18289582477882504, 0.0943981483578682, 0.050126830115914345],
            [0.18289582431316376, 0.0943981483578682, 0.00012683123350143433],
            [0.22620604932308197, 0.0694138053804636, 0.050126831978559494],
            [0.22620604559779167, 0.06941380351781845, 0.0001268293708562851],
            [0.1641575414687395, 0.06191543489694595, 0.050126828253269196],
            [0.1641575414687395, 0.06191543489694595, 0.0001268293708562851],
            [0.20746776275336742, 0.03693109005689621, 0.050126830115914345],
            [0.20746776275336742, 0.03693109005689621, 0.00012683123350143433],
            [0.14541925862431526, 0.029432721436023712, 0.050126830115914345],
            [0.14541925489902496, 0.029432719573378563, 0.00012682750821113586],
            [0.18872947990894318, 0.0044483765959739685, 0.050126828253269196],
            [0.1887294794432819, 0.0044483765959739685, 0.0001268293708562851],
            [0.12668097391724586, -0.003049992024898529, 0.050126830115914345],
            [0.12668097391724586, -0.003049992024898529, 0.0001268293708562851],
            [0.16999119520187378, -0.02803434431552887, 0.050126830115914345],
            [0.16999119520187378, -0.02803434431552887, 0.0001268293708562851],
        ]
    )

    input_dir = cfg.franka_rope.preprocess.input_dir
    output_dir = cfg.franka_rope.preprocess.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    rel = eval(str(cfg.franka_rope.preprocess.rel).title())
    rpy = eval(str(cfg.franka_rope.preprocess.rpy).title())
    flow = eval(str(cfg.franka_rope.preprocess.flow).title())

    # gravity_dir = [0, 0, -1]  # z is up in isaac sim
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
            if not i % 3 == 0:
                continue
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

            pc = np.concatenate(
                (pc, tgt_pc), axis=1
            )  # [ 1.57756746e-01  9.57879238e-03  5.00003956e-02 -7.45579600e-04 -6.01215288e-04 -4.09781933e-07]
        
            pc = np.concatenate((pc, np.full((pc.shape[0], 1), gripper_pose)), axis=1)

            delta_pos = (
                np.array(fut["Right"]["Right_target_world_position"])
                - right_target_world_pos
            )

            delta_rot = np.array(
                # mu.mul(
                #     ((fut["Right"]["Right_target_world_orientation"])),
                #     mu.inverse(
                #         (right_target_world_rot)
                #     ),  # quat_conj should close to mu.inverse(normalize_quat(right_target_world_rot))
                # )
            )

            ori = q2rmat(delta_rot)
            mat4x4 = np.eye(4)
            mat4x4[:3, :3] = ori
            mat4x4[:3, 3] = delta_pos

            # print(normalize_quat(delta_rot), rmat2q(ori)) should be close
            action = np.array(
                mat4x4
            )

            # recording["action"].append(action)
            #  state/eef_pos needs to be like [eef_pos, dir1, dir2, gravity_dir, gripper_pose]
            # dir1, dir2: first and third column of end effector’s rotation matrix (orientation)

            curr = fut
            _i = i // 3
            print(np.array(pc))
            assert not (np.isnan(np.array(pc)).any())
            assert not (np.isnan(np.array(action)).any())
            
            np.savez(
                # :02d is expected from the dataset py
                os.path.join(output_dir + rf"\01_ep{ep:06d}_view0_t{_i:02d}.npz"),
                pc=np.array(pc),
                action=np.array(action[np.newaxis, :]),
            )


if __name__ == "__main__":
    main()
