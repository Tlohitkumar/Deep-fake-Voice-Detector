import librosa, numpy as np, soundfile as sf
from config import SAMPLE_RATE
def load_audio(path, sr=SAMPLE_RATE, mono=True, trim_silence=True):
    try:
        y, orig_sr = sf.read(path)
    except:
        y, orig_sr = librosa.load(path, sr=None)
    if y.ndim>1 and mono: y=y.mean(axis=1)
    if orig_sr!=sr: y=librosa.resample(y, orig_sr, sr)
    if trim_silence: y,_=librosa.effects.trim(y)
    if np.max(np.abs(y))>0: y=y/np.max(np.abs(y))
    return y
def frame_generator(y, sr=SAMPLE_RATE, window_sec=3.0, stride_sec=1.5):
    w=int(window_sec*sr); s=int(stride_sec*sr)
    if len(y)<=w: yield y; return
    for st in range(0, len(y)-w+1, s): yield y[st:st+w]
