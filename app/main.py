import os
import sys
import traceback
# from joblib import load
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class InferenceInput(BaseModel):
    prompt: str = Field(..., example='//a program to say hello world in C', title='prompt')
class InferenceResult(BaseModel):
    code: str = Field(..., title='code output')
class InferenceResponse(BaseModel):
    error: bool = Field(..., example=False, title='Whether there is error')
    results: InferenceResult = ...
class ErrorResponse(BaseModel):
    error: bool = Field(..., example=True, title='Whether there is error')
    message: str = Field(..., example='', title='Error message')
    traceback: str = Field(None, example='', title='Detailed traceback of the error')

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-160M")
    model = AutoModelForCausalLM.from_pretrained("NinedayWang/PolyCoder-160M")
    app.package = {
        "tokenizer": tokenizer,
        "model": model
    }

@app.post("/api/generate", 
         response_model=InferenceResponse, 
         responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse}
         })
def generate(request: Request, body: InferenceInput):
    
    tokenizer = app.package["tokenizer"]
    model = app.package["model"]
    input_ids = tokenizer.encode(body.prompt, return_tensors='pt')
    res = model.generate(input_ids, max_length=200, num_beams=4)
    
    results = {
        "code": tokenizer.decode(res[0])
    }
    
    return {
        "error": False,
        "results": results
    }

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        # "torch.__version__": torch.__version__,
        # "torch.cuda.is_available()": torch.cuda.is_available(),
        # "torch.version.cuda": torch.version.cuda,
        # "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        # "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True)