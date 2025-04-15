from kornia.geometry.liegroup import Se3, So3
import torch

# x=torch.tensor([[ 2.2521,  0.0178,  0.2354, -0.0092],
#           [ 0.3776,  2.6070, -0.1922, -0.0143],
#           [-0.3752, -0.4202,  2.3786, -0.0127],
#           [-0.1122,  0.3348, -0.3346, -0.3709]])

x=torch.tensor(        [[-2.9545e-02, -6.2394e-01, -7.8091e-01, -5.6330e-02],
          [-9.9757e-01,  6.7730e-02, -1.6374e-02,  9.9494e-02],
          [ 6.3107e-02,  7.7853e-01, -6.2442e-01,  7.7267e-02],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
)
s=Se3.from_matrix(x)
print(s)
print(s.log())

noisy_actions=torch.randn((2,3,4,4))
ori_indices = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
selected_ori_actions = [noisy_actions[:, :, i, j] for i, j in ori_indices]
trans_indices = [(0, 3), (1, 3), (2, 3)]
selected_trans_actions = [noisy_actions[:, :, i, j] for i, j in trans_indices]
noisy_ori_actions = torch.stack(selected_ori_actions, dim=-1) #.repeat_interleave(num_point,dim=1) # [B,Hp,6]
noisy_trans_actions = torch.stack(selected_trans_actions, dim=-1)  #.repeat_interleave(num_point,dim=1) # [B,Hp,3]
print(len(selected_ori_actions),selected_ori_actions[0].shape)
print(noisy_ori_actions.shape,noisy_trans_actions.shape,)