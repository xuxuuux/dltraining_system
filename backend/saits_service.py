import os
import asyncio
import numpy as np
from pypots.imputation import SAITS
from sklearn.metrics import mean_absolute_error
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_DIR = "../public"
EPOCHS = 50
BATCH_SIZE = 20
MISSING_RATE = 0.1
RNG_SEED = 42

async def run_saits_with_progress(websocket, dataset_path: str):
    """训练 SAITS 并实时推送每个 epoch 的 loss & accuracy"""

    if not os.path.exists(dataset_path):
        await websocket.send_json({"status": "error", "message": f"Dataset '{dataset_path}' not found"})
        return

    arr = np.load(dataset_path)
    data = arr[np.newaxis, :, :]  # shape: (1, time, features)

    # 生成缺失数据
    rng = np.random.default_rng(seed=RNG_SEED)
    mask_missing = rng.random(data.shape) < MISSING_RATE
    data_with_nan = data.copy()
    data_with_nan[mask_missing] = np.nan

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    model = SAITS(
        n_steps=data.shape[1],
        n_features=data.shape[2],
        n_layers=3,
        d_model=64,
        n_heads=2,
        d_k=32,
        d_v=32,
        d_ffn=128,
        dropout=0.1,
        epochs=10,  # 每次循环 1 epoch
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    baseline_mae = None
    for epoch_idx in range(EPOCHS):
        # 每个 epoch 训练
        model.fit({"X": data_with_nan})
        imputed = model.impute({"X": data_with_nan})

        # 只计算缺失位置的 MAE
        if mask_missing.any():
            mae = mean_absolute_error(data[mask_missing], imputed[mask_missing])
        else:
            mae = 0.0

        if baseline_mae is None:
            baseline_mae = mae if mae > 0 else 1e-6  # 避免除零
        accuracy = max(0.0, min(1.0, 1 - mae / baseline_mae))

        # 实时推送
        await websocket.send_json({
            "epoch": epoch_idx + 1,
            "loss": float(mae),
            "accuracy": float(accuracy)
        })
        await asyncio.sleep(0.1)  # 可选，让前端能流畅更新

    # 保存模型
    model_file = os.path.join(MODEL_SAVE_DIR, "saits_model.pth")
    try:
        if hasattr(model, 'model'):
            torch.save(model.model.state_dict(), model_file)
        elif hasattr(model, 'model_'):
            torch.save(model.model_.state_dict(), model_file)
        else:
            raise AttributeError("SAITS 模型内部没有 model 属性")
        print(f"模型已保存到: {model_file}")
    except Exception as e:
        print("模型保存失败:", e)
        await websocket.send_json({"status": "error", "message": f"模型保存失败: {e}"})

    # 保存最终插补结果
    imputed_file = os.path.join(MODEL_SAVE_DIR, "imputed.npy")
    np.save(imputed_file, imputed)
    print(f"插补结果已保存到: {imputed_file}")

    # 训练完成后发送状态
    await websocket.send_json({
        "status": "done",
        "metrics": {"mae": float(mae), "accuracy": float(accuracy)},
        "model_path": model_file,
        "imputed_path": imputed_file
    })
