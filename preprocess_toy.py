import json
import numpy as np
import os
from math import sqrt
import math
import hydra
import random
# from isaacsim import SimulationApp

# simulation_app = SimulationApp({"headless": True})

# from omni.isaac.utils._isaac_utils import math as mu
from scipy.spatial.transform import Rotation as R # this operates on float64, unlike kornia which is on float32

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


# def _l2_norm(q):
#     return sqrt(sum(map(lambda x: float(x) ** 2, q)))


# def normalize_quat(q):
#     norm = _l2_norm(q)
#     q[0], q[1], q[2], q[3] = q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm
#     return q


# def q2aa(q):
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


# def quat2rpy(q):
#     q = normalize_quat(q)
#     w, x, y, z = q[0], q[1], q[2], q[3]

#     # roll
#     sinr_cosp = 2 * (w * x + y * z)
#     cosr_cosp = 1 - 2 * (x * x + y * y)
#     roll = math.atan2(sinr_cosp, cosr_cosp)

#     # pitch
#     sinp = 2 * (w * y - z * x)
#     if abs(sinp) >= 1:
#         pitch = math.copysign(math.pi / 2, sinp)
#     else:
#         pitch = math.asin(sinp)

#     # yaw
#     siny_cosp = 2 * (w * z + x * y)
#     cosy_cosp = 1 - 2 * (y * y + z * z)
#     yaw = math.atan2(siny_cosp, cosy_cosp)

#     return roll, pitch, yaw


# def rpy2quat(rpy):
#     # return euler_angles_to_quat(rpy)
#     roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

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


# def q2cols(q):
#     q = normalize_quat(q)
#     w, x, y, z = q
#     col1 = [2 * (w**2 + x**2) - 1, 2 * (x * y + w * z), 2 * (x * z - w * y)]
#     col3 = [2 * (w * y + x * z), 2 * (y * z - w * x), w**2 - x**2 - y**2 + z**2]
#     return col1, col3


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


# def rmat2q(rmat):
#     trace = np.trace(rmat)
#     if trace > 0:
#         S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
#         w = 0.25 * S
#         x = (rmat[2, 1] - rmat[1, 2]) / S
#         y = (rmat[0, 2] - rmat[2, 0]) / S
#         z = (rmat[1, 0] - rmat[0, 1]) / S
#     elif (rmat[0, 0] > rmat[1, 1]) and (rmat[0, 0] > rmat[2, 2]):
#         S = np.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2]) * 2  # S = 4 * x
#         w = (rmat[2, 1] - rmat[1, 2]) / S
#         x = 0.25 * S
#         y = (rmat[0, 1] + rmat[1, 0]) / S
#         z = (rmat[0, 2] + rmat[2, 0]) / S
#     elif rmat[1, 1] > rmat[2, 2]:
#         S = np.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2]) * 2  # S = 4 * y
#         w = (rmat[0, 2] - rmat[2, 0]) / S
#         x = (rmat[0, 1] + rmat[1, 0]) / S
#         y = 0.25 * S
#         z = (rmat[1, 2] + rmat[2, 1]) / S
#     else:
#         S = np.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1]) * 2  # S = 4 * z
#         w = (rmat[1, 0] - rmat[0, 1]) / S
#         x = (rmat[0, 2] + rmat[2, 0]) / S
#         y = (rmat[1, 2] + rmat[2, 1]) / S
#         z = 0.25 * S

#     return [w, x, y, z]


# def quat_conj(q):
#     q[1], q[2], q[3] = -q[1], -q[2], -q[3]
#     return q


# # this somehow does not match the result computed using mu.mul BUG?
# # mu.mul is used to calculate obs
# def quat_mul(q1, q2):
#     w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
#     w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

#     return [w, x, y, z]


@hydra.main(config_path="equibot/policies/configs", config_name="franka_base")
def main(cfg):

    # NEW
    tgt_pc = np.array(
        [
                        [
                            0.02616778388619423,
                            -0.17271608114242554,
                            0.060000933706760406
                        ],
                        [
                            0.026168517768383026,
                            -0.17271505296230316,
                            9.35697755721776e-07
                        ],
                        [
                            0.017845619469881058,
                            -0.23213613033294678,
                            0.05999980866909027
                        ],
                        [
                            -0.033252257853746414,
                            -0.16439391672611237,
                            0.060000352561473846
                        ],
                        [
                            0.017846353352069855,
                            -0.2321351021528244,
                            -1.8844585270016978e-07
                        ],
                        [
                            -0.03325152397155762,
                            -0.16439288854599,
                            3.523719840359263e-07
                        ],
                        [
                            -0.041574422270059586,
                            -0.2238139659166336,
                            0.05999922752380371
                        ],
                        [
                            -0.04157368838787079,
                            -0.22381293773651123,
                            -7.717716243860195e-07
                        ]
                    ]
    )

    input_dir = cfg.franka_rope.preprocess.input_dir
    output_dir = cfg.franka_rope.preprocess.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # rel = eval(str(cfg.franka_rope.preprocess.rel).title())
    # rpy = eval(str(cfg.franka_rope.preprocess.rpy).title())
    # flow = eval(str(cfg.franka_rope.preprocess.flow).title())

    gravity_dir = [0, 0, -1]  # z is up in isaac sim
    for ep, filename in enumerate(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue

        file = os.path.join(input_dir, filename)

        data = []
        with open(file, "r") as f:
            for line in f:
                data.append(json.loads(line))
                
        # tgt_pc=data[0]["Isaac Sim Data"][0]["data"]["Target_T"]["Target_pc"]

        degree = random.randint(0, 360)
        random_rotation = R.from_euler('z', degree, degrees=True).as_matrix()
        random_translation = np.array([random.randint(-2, 4),random.randint(-2, 4),0])
        
        # random_rotation=np.eye(3)
        # random_translation=np.zeros(3)
        rotated_tgt_pc=np.array([random_rotation@vector+random_translation for vector in tgt_pc])

        # to mimic saved npz with keys pc, rgb?, action, eef_pos
        for i, _fut in enumerate(data[0]["Isaac Sim Data"]):
            # if not i % 3 == 0:
            #     continue
            fut = _fut["data"]
            if i == 0:
                curr = fut
                if (
                    curr["Left"]["applied_joint_positions"][-1] < 0.025
                ):  # 0/-0.3 is closed, ~0.05 is opened
                    gripper_action = 0
                    gripper_pose = 0
                else:
                    gripper_action = 1
                    gripper_pose = 1

                continue

            # franka_joints = np.array(
            #     curr["Left"]["Left_joint_positions"]
            # )  # not exposed to the algorithm
            Left_target_world_pos = random_rotation@np.array(
                curr["Left"]["Left_target_world_position"]
            )+random_translation  # as-is
            Left_target_world_rot = np.array(
                curr["Left"]["Left_target_world_orientation"]
            )
            Left_target_world_rot=random_rotation@R.from_quat(Left_target_world_rot,scalar_first=True).as_matrix()

            t_pc = np.array(curr["Cube"]["pc"])  # as pc
            rotated_t_pc=[random_rotation@vector+random_translation for vector in t_pc]
            if (
                curr["Left"]["applied_joint_positions"][-1] < 0.025
            ):  # 0/-0.3 is closed, ~0.05 is opened
                gripper_action = 0
            else:
                gripper_action = 1

            # should be like (3460, 3)
            pc = np.array(rotated_t_pc)

            pc = np.concatenate(
                (pc, rotated_tgt_pc), axis=1
            )  # [ 1.57756746e-01  9.57879238e-03  5.00003956e-02 -7.45579600e-04 -6.01215288e-04 -4.09781933e-07]
        
            # pc = np.concatenate((pc, np.full((pc.shape[0], 1), gripper_pose)), axis=1)

            delta_pos = (
                random_rotation@np.array(fut["Left"]["Left_target_world_position"])+random_translation
                - Left_target_world_pos
            )

            # delta_rot = np.array(
            #     mu.mul(
            #         ((fut["Left"]["Left_target_world_orientation"])),
            #         mu.inverse(
            #             (Left_target_world_rot)
            #         ),  # quat_conj should close to mu.inverse(normalize_quat(Left_target_world_rot))
            #     )
            # )
            # ori = q2rmat(delta_rot)

            _q= np.array(
                fut["Left"]["Left_target_world_orientation"]
            )
            q1=random_rotation@R.from_quat(_q,scalar_first=True).as_matrix()
            q1=R.from_matrix(q1)
            _q= np.array(
                curr["Left"]["Left_target_world_orientation"]
            )
            q2=random_rotation@R.from_quat(_q,scalar_first=True).as_matrix()
            q2=R.from_matrix(q2)

            # _q=curr["Left"]["Left_target_world_orientation"]
            # q2=R.from_quat(_q,scalar_first=True)

            delta_rot=(q1*(q2.inv())).as_matrix() # 3 by 3 rot orthogonal matrix
            ori=delta_rot
            mat4x4 = np.eye(4)
            mat4x4[:3, :3] = ori
            mat4x4[:3, 3] = delta_pos
            mat4x4[3,3]=gripper_action

            # print(normalize_quat(delta_rot), rmat2q(ori)) should be close
            action = np.array(
                mat4x4
            )

            _i = i // 3
            assert not (np.isnan(np.array(pc)).any())
            assert not (np.isnan(np.array(action)).any())
            # recalculated=delta_rot@(R.from_quat(curr["Left"]["Left_target_world_orientation"],scalar_first=True).as_matrix())
            # gt=(R.from_quat(fut["Left"]["Left_target_world_orientation"],scalar_first=True).as_matrix())
            # np.testing.assert_allclose(recalculated-gt, 0, atol=1e-7)

            # tmp=delta_rot@(R.from_quat(Left_target_world_rot,scalar_first=True).as_matrix())
            # recalculated=R.from_matrix(tmp).as_quat(scalar_first=True,canonical=False)
            # gt=np.array(fut["Left"]["Left_target_world_orientation"])
            # np.testing.assert_allclose(recalculated-gt, 0, atol=1e-7)

            # recalculated=delta_pos+Left_target_world_pos
            # gt=np.array(fut["Left"]["Left_target_world_position"])
            # np.testing.assert_allclose(recalculated-gt, 0, atol=1e-7)
            #ori=R.from_quat(Left_target_world_rot,scalar_first=True).as_matrix()
            ori=Left_target_world_rot
            ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
            cols = [ori[i, j] for i, j in ori_indices]
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
            
            np.savez(
                # :02d is expected from the dataset py
                os.path.join(output_dir + rf"/01_ep{ep:06d}_view0_t{_i:02d}.npz"),
                pc=np.array(pc), # (40, 6) = (num_points, src + tgt)
                eef_pos=np.array(eef_pos), #  (13,)
                action=np.array(action[np.newaxis, :]), #  (1, 4, 4)

            )

            gripper_pose = gripper_action
            curr = fut



if __name__ == "__main__":
    main()
