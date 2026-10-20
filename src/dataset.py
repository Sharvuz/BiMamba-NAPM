import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import tools


class UCFDataset(data.Dataset):
    """
    DataLoader for UCF-Crime Video Features (CLIP ViT-B/16).
    Supports loading only normal videos (normal=True), only abnormal (normal=False),
    or all videos (for testing).
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not test_mode:
            if normal:
                self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
            else:
                self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        label = self.df.loc[index]['label']
        clip_feature = np.load(path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label


class XDDataset(data.Dataset):
    """
    DataLoader for XD-Violence Video Features (CLIP ViT-B/16).
    Supports loading only normal videos (normal=True), only abnormal (normal=False),
    or all videos (for testing).
    """
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool = False, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal

        if not test_mode:
            if normal:
                # XD-Violence normal label is 'B' or 'Normal'
                self.df = self.df.loc[self.df['label'].astype(str).str.contains('B|Normal')].reset_index(drop=True)
            else:
                self.df = self.df.loc[~self.df['label'].astype(str).str.contains('B|Normal')].reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        label = self.df.loc[index]['label']
        clip_feature = np.load(path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            return torch.tensor(clip_feature, dtype=torch.float32), label, clip_length, path, label
