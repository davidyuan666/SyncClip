from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import argparse
import logging
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()


from src.agents.slice_agent import SliceAgent
from src.agents.merge_agent import MergeAgent
from src.agents.video_analyze_agent import VideoAnalyzeAgent
from src.agents.whisper_agent import WhisperAgent
import uvicorn
import asyncio
import traceback
import os
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import Request, WebSocket, WebSocketDisconnect
import io
import wave
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的前端域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)


logger = logging.getLogger(__name__)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Worker is running"
    }

@app.post("/thread/auto_slice")
async def auto_slice(request: Request):
    try:
        data = await request.json()
        print(f"Received data: {data}")
        
        if not data.get('projectNo'):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": "Missing required field: projectNo"
                    }
                }
            )
        

        slice_agent = SliceAgent()
        logger.info("Using Python-based slice agent")
        
        # 创建任务但不等待完成
        asyncio.create_task(slice_agent.process_slice(data))
        
        # 立即返回成功接收任务的响应
        return JSONResponse(
            status_code=202,  # 使用 202 Accepted 表示请求已接受但尚未完成
            content={
                "status": "accepted",
                "data": {
                    "projectNo": data.get('projectNo'),
                    "message": "Video slice task accepted and processing in background"
                }
            }
        )
            

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing request: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "status": "error",
                    "message": f"Error accepting task: {str(e)}",
                    "details": error_traceback
                }
            }
        )


@app.post("/thread/merge")
async def auto_merge(request: Request):
    try:
        data = await request.json()
        print(f"Received data: {data}")
        
        if not data.get('projectNo'):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": "Missing required field: projectNo"
                    }
                }
            )
        

        merge_agent = MergeAgent()
        logger.info("Using Python-based merge agent")
        
        
        # 创建任务但不等待完成
        asyncio.create_task(merge_agent.process_merge(data))
        
        # 立即返回成功接收任务的响应

        return JSONResponse(
            status_code=202,  # 202 Accepted 表示请求已接受但尚未完成
            content={
                "status": "accepted",
                "data": {
                    "projectNo": data.get('projectNo'),
                    "message": "Video merge task accepted and processing in background"
                }
            }
        )
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing merge request: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": f"Error accepting merge task: {str(e)}",
                    "details": error_traceback
                }
            }
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()

    # Run FastAPI with uvicorn
    uvicorn.run(app, host='0.0.0.0', port=args.port)