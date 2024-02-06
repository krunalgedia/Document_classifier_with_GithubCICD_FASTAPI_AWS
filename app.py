from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
#from textSummarizer.pipeline.prediction import PredictionPipeline
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import shutil
from docClassify.pipeline.prediction import Predictor

#text: str = "What is Text Summarization?"

app = FastAPI()

#@app.get("/")
#async def root():
#    return {"message": "Hello World"}

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Create a temporary directory to store the uploaded image
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)

    # Save the contents of the uploaded file to a temporary file
    image_path = os.path.join(upload_dir, file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Now you have the file path of the uploaded image
    # You can use this path to process the image as needed
    predictor = Predictor(image_path)
    prediction = predictor.predict()
    # Return a response with the file path
    return {"filename": file.filename, "content_type": file.content_type, "image_path": image_path, "Prediction": prediction}


#@app.get("/train")
#async def training():
#    try:
#        os.system("python main.py")
#        return Response("Training successful !!")

    #except Exception as e:
    #    return Response(f"Error Occurred! {e}")


#@app.post("/predict")
#async def predict_route(text):
#    try:
#        obj = PredictionPipeline()
#        text = obj.predict(text)
#        return text
#    except Exception as e:
#        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)