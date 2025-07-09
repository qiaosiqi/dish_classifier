from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import shutil, os

app = FastAPI()
UPLOAD_DIR = "resnet_classifier/uploads"
PREDICT_IMG_PATH = os.path.join(UPLOAD_DIR, "latest.jpg")

os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, message: str = ""):
    return templates.TemplateResponse("index.html", {"request": request, "message": message})

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    with open(PREDICT_IMG_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse(url="/?message=图片上传成功", status_code=303)
