import torch, numpy as np
from model import SmallCNN
from utils.audio_utils import load_audio, frame_generator
from utils.feature_utils import extract_frame_level_features
from config import DEVICE, WINDOW_SEC, WINDOW_STRIDE
def load_model(p):
    m=SmallCNN(); m.load_state_dict(torch.load(p, map_location=DEVICE)); m.to(DEVICE); m.eval(); return m
def predict_file(m,fp,th=0.5):
    y=load_audio(fp); segs=list(frame_generator(y,window_sec=WINDOW_SEC,stride_sec=WINDOW_STRIDE))
    ps=[]
    for s in segs:
        f=extract_frame_level_features(s); f=(f-f.mean())/(f.std()+1e-9)
        x=torch.tensor(f).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): p=torch.softmax(m(x),1)[0,1].item()
        ps.append(p)
    if not ps: return {"prob_fake":0,"decision":"real"}
    a=float(np.mean(ps)); return {"prob_fake":a,"decision":"fake" if a>=th else "real"}
