SAMPLE_RATE = 16000
N_MFCC = 13
FRAME_LENGTH = 0.025
HOP_LENGTH = 0.01
N_MELS = 40
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SEC = 3.0
WINDOW_STRIDE = 1.5
