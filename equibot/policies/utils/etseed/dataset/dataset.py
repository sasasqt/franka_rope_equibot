import torch
import numpy as np
from pdb import set_trace as bp

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        # print('min_start',min_start,'\nmax_start',max_start)
        
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

# 17 18 0 1
def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        # read from zarr dataset

        dataset_root = np.load(dataset_path, allow_pickle=True)

        # (N, D)
        self.train_data = {
            'pts': dataset_root.item()['pts'],
            'pose':dataset_root.item()['pose'],
            'gt_action':dataset_root.item()['gt_action']
        }


        # print('type',type(dataset_root.item()['pose'][0]))

        episode_ends = dataset_root.item()['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # pad_before=obs_horizon-1,
            # pad_after=action_horizon-1)
            pad_before=0,
            pad_after=0)
        
        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]
        
        # print('buffer_start_idx',buffer_start_idx) # 7
        # print('buffer_end_idx',buffer_end_idx) # 13
        # print('sample_start_idx',sample_start_idx) # 0
        # print('sample_end_idx',sample_end_idx) # 16
        
        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon, # sequence_length
            buffer_start_idx=buffer_start_idx, 
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )
        # discard unused observations
        nsample['pts'] = nsample['pts'][:self.obs_horizon,:]
        nsample['pose'] = nsample['pose'][:self.obs_horizon,:] 
        nsample['gt_action'] = nsample['gt_action'][:self.pred_horizon,:] 
        return nsample