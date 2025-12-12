import argparse, torch, pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SmallCNN
from dataset import AudioSegmentDataset
from config import DEVICE
import torch.nn as nn
def load_metadata(p): return pd.read_csv(p).to_dict('records')
def collate(b):
    xs,ys=zip(*b)
    return torch.stack(xs), torch.stack(ys)
def train(args):
    items=load_metadata(args.csv)
    tr,va=train_test_split(items,test_size=0.15,stratify=[i['label'] for i in items],random_state=42)
    tl=DataLoader(AudioSegmentDataset(tr),batch_size=args.bs,shuffle=True,collate_fn=collate)
    vl=DataLoader(AudioSegmentDataset(va),batch_size=args.bs,shuffle=False,collate_fn=collate)
    m=SmallCNN().to(DEVICE); opt=torch.optim.Adam(m.parameters(),lr=args.lr); crit=nn.CrossEntropyLoss()
    best=1e9
    for e in range(args.epochs):
        m.train(); trl=0
        for x,y in tqdm(tl):
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            o=m(x); l=crit(o,y); l.backward(); opt.step(); trl+=l.item()*x.size(0)
        trl/=len(tl.dataset)
        m.eval(); vlss=0; cor=0; tot=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(DEVICE),y.to(DEVICE)
                o=m(x); l=crit(o,y); vlss+=l.item()*x.size(0)
                p=o.argmax(1); cor+= (p==y).sum().item(); tot+=y.size(0)
        vlss/=len(vl.dataset); acc=cor/tot
        print(e,trl,vlss,acc)
        if vlss<best:
            best=vlss; torch.save(m.state_dict(),args.save)
            print("saved",args.save)
if __name__=="__main__":
    A=argparse.ArgumentParser()
    A.add_argument("--csv"); A.add_argument("--epochs",type=int,default=5)
    A.add_argument("--bs",type=int,default=8); A.add_argument("--lr",type=float,default=1e-3)
    A.add_argument("--save",default="best_model.pth")
    train(A.parse_args())
