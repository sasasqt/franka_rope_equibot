import torch
import torch.nn.functional as F

def batched_gram_schmidt_columns(batched_2cols_flattened):
    # first two cols
    b1 = batched_2cols_flattened[:, 0:3]
    e1 = F.normalize(b1, p=2, dim=-1)  # Normalize u1

    b2 = batched_2cols_flattened[:, 3:6]
    proj_u2_on_e1 = (torch.sum(b2 * e1, dim=-1, keepdim=True) * e1)
    b2 = b2 - proj_u2_on_e1
    e2 = F.normalize(b2, p=2, dim=-1)  # Normalize v2
 
    e3 = torch.cross(e1, e2, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)  # Shape: [batch_size, 3, 3]
    return R


def main():
    matrix = torch.rand(3, 3)
    print(matrix)
    # Select the first two columns
    result = [(0, 0), (1,0), (2,0), (0, 1), (1,1), (2,1)] # first two cols
    selected_ori_actions = [matrix[i, j] for i, j in result]
    print(selected_ori_actions)
    print(batched_gram_schmidt_columns(torch.tensor([[1,0,0,0,0,1]],dtype=torch.float32)))
    # tensor([[[ 1.,  0.,  0.],
    #         [ 0.,  0., -1.],
    #         [ 0.,  1.,  0.]]])
if __name__ == '__main__':
    main()
