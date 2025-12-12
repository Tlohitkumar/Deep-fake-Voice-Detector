from fastapi import FastAPI, UploadFile, File
import uvicorn, os, shutil
from inference import load_model, predict_file
MODEL_PATH="best_model.pth"; TMP="tmp_uploads"
os.makedirs(TMP,exist_ok=True)
app=FastAPI(); model=load_model(MODEL_PATH)
@app.post("/predict")
async def pred(file:UploadFile=File(...)):
    ext=os.path.splitext(file.filename)[1].lower()
    if ext not in [".wav",".mp3",".flac",".m4a",".ogg"]:
        return {"error":"bad file"}
    p=os.path.join(TMP,file.filename)
    with open(p,"wb") as f: shutil.copyfileobj(file.file,f)
    r=predict_file(model,p); os.remove(p); return r
if __name__=="__main__": uvicorn.run(app,host="0.0.0.0",port=8000)
