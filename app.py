from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid

# 导入你的预测函数
from predict import predict_image

app = FastAPI()

# 临时存储上传图片的目录
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 保存上传文件
    file_ext = file.filename.split('.')[-1]
    temp_filename = f"{uuid.uuid4().hex}.{file_ext}"
    temp_path = UPLOAD_DIR / temp_filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 进行预测
        result = predict_image(str(temp_path))
        return JSONResponse(content={"success": True, "result": result})
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})
    finally:
        # 删除临时文件
        temp_path.unlink(missing_ok=True)
