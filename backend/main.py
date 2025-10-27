from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from saits_service import run_saits_with_progress
import os

app = FastAPI(title="SAITS Training API")

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 static 文件夹，浏览器可通过 /static/... 访问
# backend/static/pems.npy
# backend/static/saits_model/imputed.npy
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket 训练接口
@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    try:
        # dataset 默认使用 static/pems.npy
        dataset_path = os.path.join("static", "pems.npy")
        if not os.path.exists(dataset_path):
            await websocket.send_json({"status": "error", "message": f"Dataset '{dataset_path}' not found"})
            return

        await run_saits_with_progress(websocket, dataset_path=dataset_path)

    except WebSocketDisconnect:
        print("前端断开连接")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})


