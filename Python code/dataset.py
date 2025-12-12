import torch, numpy as np
from torch.utils.data import Dataset
from utils.audio_utils import load_audio, frame_generator
from utils.feature_utils import extract_frame_level_features
from config import SAMPLE_RATE, WINDOW_SEC, WINDOW_STRIDE
class AudioSegmentDataset(Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it=self.items[i]; y=load_audio(it['path'])
        segs=list(frame_generator(y, SAMPLE_RATE, WINDOW_SEC, WINDOW_STRIDE))
        seg=segs[0] if segs else y
        f=extract_frame_level_features(seg)
        f=(f-f.mean())/(f.std()+1e-9)
        x=torch.tensor(f).float().unsqueeze(0)
        y=torch.tensor(it['label']).long()
        return x,y
