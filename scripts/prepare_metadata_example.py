import os, argparse, pandas as pd
A=argparse.ArgumentParser()
A.add_argument("--real_dir"); A.add_argument("--fake_dir"); A.add_argument("--out_csv",default="metadata.csv")
a=A.parse_args()
rows=[]
for d,l in [(a.real_dir,0),(a.fake_dir,1)]:
    for p in os.listdir(d):
        if p.lower().endswith((".wav",".mp3",".flac",".m4a",".ogg")):
            rows.append({"path":os.path.join(d,p),"label":l})
pd.DataFrame(rows).to_csv(a.out_csv,index=False)
print("saved",a.out_csv)
