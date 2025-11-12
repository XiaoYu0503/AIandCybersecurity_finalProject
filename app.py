import io
import os
import tempfile
from typing import Tuple, List, Any

import numpy as np
import streamlit as st
from PIL import Image
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None  # 讓 UI 顯示提示

st.set_page_config(page_title="平交道障礙物即時辨識 (ONNX)", layout="wide")
st.title("平交道障礙物辨識 (Streamlit / ONNX)")
st.caption("請上傳已轉換的 YOLOv7 ONNX 權重（.onnx）。信心度低於 0.7 的框不顯示。若仍只有 .pt，請先在本地轉換再上傳。")

CONF_THRESH = 0.7
INPUT_SIZE = 640  # 典型 YOLOv7 輸入尺寸，若訓練時不同請調整

def _assumption_notice():
    st.info(
        "本重寫版本改用 ONNXRuntime 以避免 Python 3.13 下 torch 安裝問題。\n"
        "請先在本地把 YOLOv7 的 best.pt 轉換成 .onnx：\n"
        "1. 取得 YOLOv7 原始倉庫。\n"
        "2. 在該倉庫使用 export 腳本 (示例)： python export.py --weights best.pt --grid --end2end --simplify --topk-all 100 --device cpu \n"
        "3. 於此頁面上傳輸出的 .onnx 檔案。"
    )

@st.cache_resource(show_spinner=True)
def load_session(weights_path: str) -> object:
    if ort is None:
        raise RuntimeError("onnxruntime 未安裝，請確認 requirements.txt 已包含 onnxruntime。")
    # 使用 CPU 執行
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(weights_path, so, providers=["CPUExecutionProvider"])

def letterbox(image: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # 填充到方形
    pad_w, pad_h = new_shape - nw, new_shape - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    boxed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return boxed, scale, (left, top)

def preprocess(image: np.ndarray) -> np.ndarray:
    boxed, scale, (left, top) = letterbox(image, INPUT_SIZE)
    img = boxed.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # BCHW
    return img, scale, left, top

def postprocess(outputs: List[np.ndarray], original: np.ndarray, scale: float, pad_left: int, pad_top: int) -> np.ndarray:
    """根據 YOLO ONNX 標準輸出格式 (batch, num_boxes, 85) 或 (num_boxes, 85)。
    為簡化：不使用 NMS，依據物件 conf 直接過濾，若多框可能重疊。"""
    if len(outputs) == 0:
        return original
    out = outputs[0]
    if out.ndim == 3:
        out = out[0]
    # 判斷列長度 (>5 表示含 class scores)
    h0, w0 = original.shape[:2]
    annotated = original.copy()
    for det in out:
        if det.shape[0] < 5:
            continue
        x1, y1, x2, y2, obj_conf = det[:5]
        cls_conf = 0.0
        if det.shape[0] > 5:
            cls_scores = det[5:]
            if cls_scores.size > 0:
                cls_conf = float(cls_scores.max())
        conf = float(obj_conf) * (cls_conf if cls_conf > 0 else 1.0)
        if conf < CONF_THRESH:
            continue
        # 將座標還原回原圖尺度
        # 先移除 padding，再除以 scale
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        # Clamp
        x1 = max(0, min(w0 - 1, x1))
        y1 = max(0, min(h0 - 1, y1))
        x2 = max(0, min(w0 - 1, x2))
        y2 = max(0, min(h0 - 1, y2))
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return annotated

# ======= 模型權重選擇（側邊欄） =======
st.sidebar.header("模型設定 (ONNX)")
uploaded_w = st.sidebar.file_uploader("上傳 ONNX 權重 (.onnx)", type=["onnx"], help="請先在本地把 best.pt 轉為 .onnx")
weights_path: str | None = None
if uploaded_w is not None:
    wtmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
    wtmp.write(uploaded_w.read())
    wtmp.flush()
    weights_path = wtmp.name
else:
    # 提供本地開發測試：若同資料夾存在 model.onnx 自動載入
    candidate = os.path.join(os.path.dirname(__file__), "model.onnx")
    if os.path.exists(candidate):
        weights_path = candidate

session = None
if ort is None:
    st.error("onnxruntime 套件未安裝，請確認 requirements.txt 已加入 onnxruntime。")
elif weights_path is None:
    st.sidebar.warning("尚未提供 ONNX 權重檔，請上傳 .onnx 後再進行推論。")
    _assumption_notice()
else:
    try:
        session = load_session(weights_path)
        st.success(f"已載入 ONNX 模型：{os.path.basename(weights_path)}")
    except Exception as e:
        st.error(f"ONNX 模型載入失敗：{e}")
        _assumption_notice()

def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def predict_image(np_rgb: np.ndarray, session: object) -> np.ndarray:
    inp, scale, left, top = preprocess(np_rgb)
    inputs = {session.get_inputs()[0].name: inp}
    outputs = session.run(None, inputs)
    annotated = postprocess(outputs, np_rgb, scale, left, top)
    return annotated

def predict_video_to_file(in_path: str, out_path: str, session: object, progress_placeholder) -> Tuple[int, int, float]:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("無法開啟上傳的影片檔案。")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    progress = st.progress(0)
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb = predict_image(frame_rgb, session)
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            frame_idx += 1
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()
        writer.release()
        progress.empty()
    return (w, h, fps)

# ======= UI：上傳與推論 =======
st.subheader("圖片推論")
image_file = st.file_uploader("上傳圖片 (jpg/png/webp)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False, disabled=(session is None))

col1, col2 = st.columns(2)
with col1:
    if image_file is not None and session is not None:
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="原始圖片", use_column_width=True)
with col2:
    if image_file is not None and session is not None:
        with st.spinner("模型推論中…"):
            rgb = _pil_to_np_rgb(img)
            annotated = predict_image(rgb, session)
            st.image(annotated, caption=f"推論結果（conf ≥ {CONF_THRESH:.2f}）", use_column_width=True)

st.markdown("---")
st.subheader("影片推論")
video_file = st.file_uploader("上傳影片 (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False, disabled=(session is None))

if video_file is not None and session is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as src_tmp:
        src_tmp.write(video_file.read())
        src_path = src_tmp.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst_tmp:
        out_path = dst_tmp.name
    st.write("開始處理影片，請稍候（依影片長度而定）…")
    ph = st.empty()
    with st.spinner("影片推論中…"):
        w, h, fps = predict_video_to_file(src_path, out_path, session, ph)
    st.success("影片處理完成！")
    with open(out_path, "rb") as f:
        st.video(f.read())
    with open(out_path, "rb") as f:
        st.download_button(
            label="下載處理後影片",
            data=f,
            file_name="predicted.mp4",
            mime="video/mp4"
        )
    try:
        os.remove(src_path)
    except Exception:
        pass

st.markdown("---")
st.caption(
    "本版本改為 ONNXRuntime 推論，避免 torch 在 Python 3.13 的相容性問題；僅顯示 conf ≥ 0.7。若需要使用原始 .pt，請先離線轉換為 .onnx 後再上傳。"
)

