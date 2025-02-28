import numpy as np
import torch


def to_torch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def rotate_around_z(
    points,
    angle_rad=0.0,
    center=np.array([0.0, 0.0, 0.0]),
    scale=np.array([1.0, 1.0, 1.0]),
):
    # Check if the input points have the correct shape (N, 3)
    assert (len(points.shape) == 1 and len(points) == 3) or points.shape[-1] == 3
    p_shape = points.shape
    points = points.reshape(-1, 3) - center[None]

    # Create the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Apply the rotation to all points using matrix multiplication
    rotated_points = np.dot(points, rotation_matrix.T) * scale[None] + center[None]
    rotated_points = rotated_points.reshape(p_shape)

    return rotated_points


def get_env_class(env_name):
    if env_name == "fold":
        from equibot.envs.sim_mobile.folding_env import FoldingEnv
        return FoldingEnv
    elif env_name == "cover":
        from equibot.envs.sim_mobile.covering_env import CoveringEnv
        return CoveringEnv
    elif env_name == "close":
        from equibot.envs.sim_mobile.closing_env import ClosingEnv
        return ClosingEnv
    else:
        raise ValueError()


def get_dataset(cfg, mode="train",valid=False):
    from equibot.policies.datasets.dataset import BaseDataset
    if valid:
        return BaseDataset(cfg.data.valid_dataset, mode)
    return BaseDataset(cfg.data.dataset, mode)


def get_agent(agent_name):
    if agent_name == "dp":
        from equibot.policies.agents.dp_agent import DPAgent
        return DPAgent
    elif agent_name == "equibot":
        from equibot.policies.agents.equibot_agent import EquiBotAgent
        return EquiBotAgent
    elif agent_name == "pfc_equibot":
        from equibot.policies.agents.pfc_equibot_agent import PfcEquiBotAgent
        return PfcEquiBotAgent
    elif agent_name == "wtf":
        from equibot.policies.agents.vn_pc_vn_obs_dp_inv2equi_agent import WTFAgent
        return WTFAgent
    elif agent_name == "etseed":
        from equibot.policies.agents.etseed_agent import ETSEEDAgent
        return ETSEEDAgent
    
    else:
        raise ValueError(f"Agent with name [{agent_name}] not found.")
