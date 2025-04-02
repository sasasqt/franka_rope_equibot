import os
from equibot.policies.utils.etseed.model.se3_transformer.equinet import SE3ManiNet_Invariant_Separate, SE3ManiNet_Equivariant_Separate, SE3ManiNet_Fused, SE3VisionNet, SE3VisionNet_Hierarchical
from equibot.policies.utils.diffusion.conditional_unet1d import ConditionalUnet1D
from equibot.policies.utils.etseed.utils.group_utils import bgs, bgdR
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import tqdm
from pdb import set_trace as bp
import math

#! Check invariant model
def check_invariant_model(num_trials=10, threshold=0.01):
    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)

        rotated_pts = rot.apply(pts) + np.random.rand(3)
        xyz = np.stack([pts, rotated_pts], axis=0)

        feature1 = np.random.rand(100,7)
        feature2=np.random.rand(100,3)
        feature3=np.random.rand(100,3)
        
        feature=np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=np.concatenate([feature1,rot.apply(feature2),rot.apply(feature3)],axis=-1)
        feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 13)
                        
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3ManiNet_Invariant_Separate().cuda()
        result = model(data)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0] - result_np[1]
        print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)
    print('Inv test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)

#! Check equivariant model
#! Although all the rotations written in the code are right multiplications, the actual physical meaning is left multiplication.
def check_equivariant_model(num_trials=10, threshold=np.pi/180*1):

    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)
        trans=np.random.rand(3)

        rotated_pts = rot.apply(pts) + trans

        # undo_tran=result_np[1][:,:3,3]-trans
        # undo_rot=np.einsum('ij,bjk->bjk',rot.inv().as_matrix(),result_np[1][:,:3,:3])
        # undo_result_np1=np.tile(np.eye(4),(pts.shape[0], 1, 1))
        # undo_result_np1[:,:3,3]=undo_tran
        # undo_result_np1[:,:3,:3]=undo_rot

        # difference = result_np[0] - undo_result_np1

        xyz = np.stack([pts, rotated_pts], axis=0)

        feature1 = np.random.rand(100,7)
        feature2=np.random.rand(100,3)
        feature3=np.random.rand(100,3)
        
        feature=np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=np.concatenate([feature1,rot.apply(feature2),rot.apply(feature3)],axis=-1)
        feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 13)
                        
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3ManiNet_Equivariant_Separate().cuda()
        result = model(data,return_raw=True)
        result = np.concatenate((result['pos'].detach().cpu().numpy(), result['ori'].detach().cpu().numpy()), axis=-1)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i])
        result_np = np.array(result_np)

        undo_rot=rot.inv().apply(result_np[1][:,:3].reshape(-1,3))

         
        # undo_tran=result_np[1][:,:3,3]-trans
        # undo_rot=np.einsum('ij,bjk->bjk',rot.inv().as_matrix(),result_np[1][:,:3,:3])
        # undo_result_np1=np.tile(np.eye(4),(pts.shape[0], 1, 1))
        # undo_result_np1[:,:3,3]=undo_tran
        # undo_result_np1[:,:3,:3]=undo_rot

        pos_diff = result_np[0][:,:3].reshape(-1,3) - undo_rot
        print("pos",pos_diff)

        rot_diff = result_np[0][:,3:].reshape(-1,3) - result_np[1][:,3:].reshape(-1,3)
        print("rot",rot_diff)
        
        if (np.allclose(pos_diff, 0.0, atol=threshold) and np.allclose(rot_diff, 0.0, atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)
    print('Equiv test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)


    # total_trials = num_trials
    # success_count = 0
    # failure_count = 0
    # zero_count = 0
    
    # total_trans_dist = 0
    # total_rot_dist = 0
    # total_real_dist = 0
    
    # for trial in tqdm.tqdm(range(total_trials)):
    #     rot = R.random()
    #     pts = np.random.rand(100,3)
    #     num_points = pts.shape[0]
    #     trans = np.random.rand(3) * 10
    #     T = np.eye(4)
    #     T[:3,:3] = rot.as_matrix()
    #     T[:3,3] = trans
    #     rotated_pts = pts @ rot.as_matrix().T + trans
    #     xyz = np.stack([pts, rotated_pts], axis=0)

    #     feature1 = np.random.rand(100,7)
    #     feature2=np.random.rand(100,3)
    #     feature3=np.random.rand(100,3)
        
    #     feature=np.concatenate([feature1,feature2,feature3],axis=-1)
    #     rot_feature=np.concatenate([feature1,rot.apply(feature2),rot.apply(feature3)],axis=-1)
    #     feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 13)
        
    #     xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
    #     feature = torch.tensor(feature, dtype=torch.float32).cuda()
    #     data = {}
    #     data['xyz'] = xyz
    #     data['feature'] = feature

    #     #! model output
    #     result_np = []
    #     model = SE3ManiNet_Equivariant_Separate().cuda()
    #     result = model(data)
    #     for i in range(len(result)):
    #         result_np.append(result[i].view(-1,4,4).detach().cpu().numpy())
    #     # phi(Rx)
    #     phi_Tx = result_np[1]
    #     # Rphi(x)
    #     phi_x = result_np[0]
    #     Tphi_x = np.eye(4)
    #     Tphi_x=np.tile(Tphi_x, (pts.shape[0], 1, 1))

    #     print(phi_x.shape, Tphi_x.shape)
    #     Tphi_x[:,:3,:3] = phi_x[:,:3,:3] @ rot.as_matrix().T
    #     Tphi_x[:3,3] = phi_x[:3,3] + trans
    #     if (np.allclose(phi_Tx, np.zeros_like(phi_Tx), atol=1e-1)):
    #         zero_count += 1
    #         # print('zero')
    #         continue
    #     geo_dist = bgdR(torch.tensor(phi_Tx)[:3,:3].unsqueeze(0), torch.tensor(Tphi_x[:3,:3]).unsqueeze(0)).item()
    #     trans_dist = np.linalg.norm(phi_Tx[:3,3] - Tphi_x[:3,3])
        
    #     real_dist = math.sqrt(geo_dist ** 2 + trans_dist ** 2)
        
    #     total_rot_dist = total_rot_dist + geo_dist
    #     total_trans_dist = total_trans_dist + trans_dist
    #     total_real_dist = total_real_dist + real_dist
    #     print(f"real_dist: {real_dist}")
    #     if (geo_dist<threshold and trans_dist<threshold):
    #         success_count += 1
    #     else:
    #         failure_count += 1
    # print('Equiv test pass rate:', int(success_count/total_trials*100), '%')
    # print("geo_dist:",total_rot_dist/total_trials,"trans_dist:",total_trans_dist/total_trials,"real_dist:",total_real_dist/total_trials)


# this assert num_fib_in = [7,2]
def check_fused_model(num_trials=10, threshold=0.01):

    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)
        trans=np.random.rand(3)-2

        rotated_pts = rot.apply(pts) + trans

        # undo_tran=result_np[1][:,:3,3]-trans
        # undo_rot=np.einsum('ij,bjk->bjk',rot.inv().as_matrix(),result_np[1][:,:3,:3])
        # undo_result_np1=np.tile(np.eye(4),(pts.shape[0], 1, 1))
        # undo_result_np1[:,:3,3]=undo_tran
        # undo_result_np1[:,:3,:3]=undo_rot

        # difference = result_np[0] - undo_result_np1

        xyz = np.stack([pts, rotated_pts], axis=0)

        feature1 = np.random.rand(100,7)-0.8
        feature2=np.random.rand(100,3)-0.8
        feature3=np.random.rand(100,3)-0.8
        
        feature=np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=np.concatenate([feature1,rot.apply(feature2),rot.apply(feature3)],axis=-1)
        feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 13)
                        
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3ManiNet_Fused_Separate().cuda()

        result = model(data,num_point=100,return_raw=False,Inv=True)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0] - result_np[1]
        #print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)



        result = model(data,num_point=100,return_raw=True,Inv=False)
        result = np.concatenate((result['pos'].detach().cpu().numpy(), result['ori'].detach().cpu().numpy()), axis=-1)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i])
        result_np = np.array(result_np)

        undo_rot=rot.inv().apply(result_np[1][:,:3].reshape(-1,3))

        pos_diff = result_np[0][:,:3].reshape(-1,3) - undo_rot
        print("pos",pos_diff)

        rot_diff = result_np[0][:,3:].reshape(-1,3) - result_np[1][:,3:].reshape(-1,3)
        print("rot",result_np[0][:,3:].reshape(-1,3) - result_np[1][:,3:].reshape(-1,3))
        
        if (np.allclose(pos_diff, 0.0, atol=threshold) and np.allclose(rot_diff, 0.0, atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)
    print('Fused test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)



# this assert num_fib_in = [7,2]
def check_fused_model2(num_trials=10, threshold=0.01):

    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)
        trans=np.random.rand(3)-2

        rotated_pts = rot.apply(pts) + trans

        # undo_tran=result_np[1][:,:3,3]-trans
        # undo_rot=np.einsum('ij,bjk->bjk',rot.inv().as_matrix(),result_np[1][:,:3,:3])
        # undo_result_np1=np.tile(np.eye(4),(pts.shape[0], 1, 1))
        # undo_result_np1[:,:3,3]=undo_tran
        # undo_result_np1[:,:3,:3]=undo_rot

        # difference = result_np[0] - undo_result_np1


        feature1 = np.random.rand(100,7)-0.8
        feature2=np.random.rand(100,3)-0.8
        feature3=np.random.rand(100,3)-0.8
        
        feature=np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=np.concatenate([feature1,rot.apply(feature2),rot.apply(feature3)],axis=-1)

        data = {}
        
        np_xyz = np.stack([pts], axis=0)
        tensor_xyz = torch.tensor(np_xyz, dtype=torch.float32).cuda()
        np_feature = np.stack([feature], axis=0)  # (2, 100, 13)     
        tensor_feature = torch.tensor(np_feature, dtype=torch.float32).cuda()

        data['xyz'] = tensor_xyz
        data['feature'] = tensor_feature
        model = SE3ManiNet_Fused_Separate().cuda()

        result = model(data,num_point=100,return_raw=False,Inv=True)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())

        np_xyz = np.stack([rotated_pts], axis=0)
        tensor_xyz = torch.tensor(np_xyz, dtype=torch.float32).cuda()
        np_feature = np.stack([rot_feature], axis=0)  # (2, 100, 13)
        tensor_feature = torch.tensor(np_feature, dtype=torch.float32).cuda()

        data['xyz'] = tensor_xyz
        data['feature'] = tensor_feature
        result = model(data,num_point=100,return_raw=False,Inv=True)
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())

        result_np = np.array(result_np)
        difference = result_np[0] - result_np[1]
        print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            print('!!!!!!!Invariant model is wrong')
            success_record.append(0)


        np_xyz = np.stack([pts], axis=0)
        tensor_xyz = torch.tensor(np_xyz, dtype=torch.float32).cuda()
        np_feature = np.stack([feature], axis=0)  # (2, 100, 13)     
        tensor_feature = torch.tensor(np_feature, dtype=torch.float32).cuda()

        data['xyz'] = tensor_xyz
        data['feature'] = tensor_feature
        result = model(data,num_point=100,return_raw=True,Inv=False)
        result = np.concatenate((result['pos'].detach().cpu().numpy(), result['ori'].detach().cpu().numpy()), axis=-1)
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i])


        np_xyz = np.stack([rotated_pts], axis=0)
        tensor_xyz = torch.tensor(np_xyz, dtype=torch.float32).cuda()
        np_feature = np.stack([rot_feature], axis=0)  # (2, 100, 13)
        tensor_feature = torch.tensor(np_feature, dtype=torch.float32).cuda()

        data['xyz'] = tensor_xyz
        data['feature'] = tensor_feature
        result = model(data,num_point=100,return_raw=True,Inv=False)
        result = np.concatenate((result['pos'].detach().cpu().numpy(), result['ori'].detach().cpu().numpy()), axis=-1)
        for i in range(len(result)):
            result_np.append(result[i])
        result_np = np.array(result_np)

        undo_rot=rot.inv().apply(result_np[1][:,:3].reshape(-1,3))

        pos_diff = result_np[0][:,:3].reshape(-1,3) - undo_rot
        print("pos",pos_diff)

        rot_diff = result_np[0][:,3:].reshape(-1,3) - result_np[1][:,3:].reshape(-1,3)
        print("rot",result_np[0][:,3:].reshape(-1,3) - result_np[1][:,3:].reshape(-1,3))
        
        if (np.allclose(pos_diff, 0.0, atol=threshold) and np.allclose(rot_diff, 0.0, atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            print('!!!!!!!equivariant model is wrong')
            success_record.append(0)
    print('Fused test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)



def check_vision_model(num_trials=10, threshold=0.01):

    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)
        trans=np.random.rand(3)

        rotated_pts = rot.apply(pts) + trans
        xyz = np.stack([pts, rotated_pts], axis=0)

        #feature1 = np.random.rand(100,7)
        feature2=np.random.rand(100,3)
        feature3=np.random.rand(100,3)
        
        feature=feature2 #np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=rot.apply(feature2)
        feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 6)
                        
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3VisionNet().cuda()


        raw = model(data,return_raw=True)
        result=raw['weights']
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0] - result_np[1]
        print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)


        result = raw['global_feat']
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0].reshape(-1,3) - rot.inv().apply(result_np[1].reshape(-1,3))
        print("difference",difference)
        
        if (np.allclose(result_np[0].reshape(-1,3), rot.inv().apply(result_np[1].reshape(-1,3)), atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)

    print('Vision test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)





def check_hierarchical_vision_model(num_trials=10, threshold=0.01):

    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)
        trans=np.random.rand(3)

        rotated_pts = rot.apply(pts) + trans
        xyz = np.stack([pts, rotated_pts], axis=0)

        #feature1 = np.random.rand(100,7)
        feature2=np.random.rand(100,3)
        feature3=np.random.rand(100,3)
        
        feature=feature2 #np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=rot.apply(feature2)
        feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 6)
                        
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3VisionNet_Hierarchical().cuda()


        raw,global_feat = model(data,return_raw=True)
        result=raw['weights']
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0] - result_np[1]
        print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)


        result=raw['global_feat']
        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0].reshape(-1,3) - rot.inv().apply(result_np[1].reshape(-1,3))
        print("difference",difference)
        
        if (np.allclose(result_np[0].reshape(-1,3), rot.inv().apply(result_np[1].reshape(-1,3)), atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)

    print('Vision test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)



def check_condunet1d(num_trials=10, threshold=0.01):
    success_record = []
    for test_inv_trial in tqdm.tqdm(range(num_trials)):
        rot = R.random()
        pts = np.random.rand(100,3)

        rotated_pts = rot.apply(pts) + np.random.rand(3)
        xyz = np.stack([pts, rotated_pts], axis=0)

        feature1 = np.random.rand(100,7)
        feature2=np.random.rand(100,3)
        feature3=np.random.rand(100,3)
        
        feature=np.concatenate([feature1,feature2,feature3],axis=-1)
        rot_feature=np.concatenate([feature1,rot.apply(feature2),rot.apply(feature3)],axis=-1)
        feature = np.stack([feature, rot_feature], axis=0)  # (2, 100, 13)
                        
        xyz = torch.tensor(xyz, dtype=torch.float32).cuda()
        feature = torch.tensor(feature, dtype=torch.float32).cuda()
        data = {}
        
        data['xyz'] = xyz
        data['feature'] = feature
        model = SE3ManiNet_Invariant_Separate().cuda()
        result = model(data)

        arot = rot#R.random()
        apts = np.random.rand(100,3)

        arotated_pts = rot.apply(apts) + np.random.rand(3)
        axyz = np.stack([apts, arotated_pts], axis=0)

        afeature1 = np.random.rand(100,7)
        afeature2=np.random.rand(100,3)
        afeature3=np.random.rand(100,3)
        
        afeature=np.concatenate([afeature1,afeature2,afeature3],axis=-1)
        arot_feature=np.concatenate([afeature1,arot.apply(afeature2),arot.apply(afeature3)],axis=-1)
        afeature = np.stack([afeature, arot_feature], axis=0)  # (2, 100, 13)
                        
        axyz = torch.tensor(axyz, dtype=torch.float32).cuda()
        afeature = torch.tensor(afeature, dtype=torch.float32).cuda()
        adata = {}
        
        adata['xyz'] = axyz
        adata['feature'] = afeature
        amodel = SE3ManiNet_Invariant_Separate().cuda()
        aresult = amodel(adata)


        result_np = []
        for i in range(len(result)):
            result_np.append(result[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        
        difference = result_np[0] - result_np[1]
        # print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            # print('Invariant model is correct')
            success_record.append(1)
        else:
            # print('!!!!!!!Invariant model is wrong')
            success_record.append(0)


        unet = ConditionalUnet1D(
            input_dim=16,
            diffusion_step_embed_dim=1600, #hierarchy_layers*output_type_1_feat*3
            global_cond_dim=1600,
            cond_predict_scale=True
        ).cuda()

        k = torch.randint(0, 1, (2,), device='cuda')

        # aresult=torch.rand((100,16), device='cuda')
        # aresult=aresult.unsqueeze(0).unsqueeze(0).expand(result.shape[0],result.shape[1],-1)
        uout=unet(result.view(result.shape[0],result.shape[1],-1),k,global_cond=aresult.view(aresult.shape[0],-1))
        result_np = []
        for i in range(len(uout)):
            result_np.append(uout[i].detach().cpu().numpy())
        result_np = np.array(result_np)
        difference = result_np[0] - result_np[1]
        print("difference",difference)
        
        if (np.allclose(result_np[0], result_np[1], atol=threshold)):
            print('condunet1d Invariant model is correct')
            success_record.append(1)
        else:
            print('!!!!!!!condunet1d Invariant model is wrong')
            success_record.append(0)

    print('Inv test pass rate:', np.mean(success_record)*100, '%')
    return np.mean(success_record)




# check_invariant_model(num_trials=100, threshold=0.01)
# check_equivariant_model(num_trials=100,threshold=0.01)
# check_fused_model(num_trials=100,threshold=0.000001)
# check_fused_model2(num_trials=100,threshold=0.000001)
# check_vision_model(num_trials=10,threshold=0.01)
# check_hierarchical_vision_model(num_trials=10,threshold=0.01)

check_condunet1d(num_trials=10,threshold=0.01)

#!! test Schimidt
# a = torch.tensor(np.random.rand(10000,3,2)).cuda()
# Sch_a = bgs(a).detach().cpu().numpy()
# for i in range(10000):
#     # print(np.linalg.det(Sch_a[i]))
#     if(not np.allclose(np.linalg.det(Sch_a[i]), 1.0)):
#         print('wrong')
#         bp()