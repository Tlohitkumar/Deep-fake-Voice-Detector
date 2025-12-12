import numpy as np, librosa
from config import SAMPLE_RATE, N_MFCC
def extract_frame_level_features(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    hop=int(0.01*sr); win=int(0.025*sr)
    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc,hop_length=hop,n_fft=512,win_length=win)
    d=librosa.feature.delta(mfcc); d2=librosa.feature.delta(mfcc,order=2)
    sc=librosa.feature.spectral_centroid(y=y,sr=sr,hop_length=hop)
    scon=librosa.feature.spectral_contrast(y=y,sr=sr,hop_length=hop)
    z=librosa.feature.zero_crossing_rate(y,hop_length=hop)
    r=librosa.feature.rms(y=y,hop_length=hop)
    try:
        f0,_,_=librosa.pyin(y,sr=sr,frame_length=win,hop_length=hop,
            fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'))
        f0=np.nan_to_num(f0).reshape(1,-1)
    except: f0=np.zeros((1,mfcc.shape[1]))
    feat=np.vstack([mfcc,d,d2,sc,scon,z,r,f0])
    return np.nan_to_num(feat)
def aggregate_stats(M):
    m=M.mean(1); s=M.std(1)
    sk=np.nanmean(((M-m[:,None])**3),1)/(s+1e-9)**3
    kt=np.nanmean(((M-m[:,None])**4),1)/(s+1e-9)**4
    return np.nan_to_num(np.concatenate([m,s,sk,kt]))
