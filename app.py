import io
import os
import tempfile
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image
import cv2

st.set_page_config(page_title="平交道障礙物即時辨識", layout="wide")
st.title("平交道障礙物辨識 (Streamlit)")
st.caption("上傳圖片或影片，系統將以選定模型權重進行推論（建議 YOLOv7）；信心度低於 0.7 的框將不顯示。")

CONF_THRESH = 0.7

@st.cache_resource(show_spinner=True)
def load_model(weights_path: str) -> Tuple[str, object]:
    """載入 YOLO 模型（依照指定的權重路徑）。
    優先使用 YOLOv7（torch.hub），失敗則嘗試 ultralytics，最後再嘗試 YOLOv5（torch.hub）。
    回傳 (backend, model)；backend ∈ { 'yolov7', 'ultralytics', 'yolov5' }。
    """
    # 1) 優先：YOLOv7 via torch.hub（需要可連網以抓取 repo hub）
    try:
        import torch  # type: ignore
        model = torch.hub.load(
            'WongKinYiu/yolov7', 'custom', path=weights_path, trust_repo=True, force_reload=False
        )
        try:
            model.to('cpu')
        except Exception:
            pass
        # 設定信心度門檻，確保低於門檻者不繪製
        try:
            model.conf = CONF_THRESH
        except Exception:
            pass
        return ("yolov7", model)
    except Exception as e_y7:
        y7_err = e_y7

    # 簡化：若 YOLOv7 失敗則直接拋錯，避免 hub 套件路徑衝突
    raise RuntimeError(
        "模型載入失敗（YOLOv7）。請確認權重檔為 YOLOv7 格式，且環境可連網以載入 torch.hub。\n"
        f"YOLOv7 錯誤: {y7_err}"
    )

# ======= 模型權重選擇（側邊欄） =======
st.sidebar.header("模型設定")
uploaded_w = st.sidebar.file_uploader("上傳模型權重 (.pt)", type=["pt"], help="未上傳時將不載入模型")
weights_path: str | None = None
if uploaded_w is not None:
    # 把上傳的 pt 寫到會話暫存檔
    wtmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    wtmp.write(uploaded_w.read())
    wtmp.flush()
    weights_path = wtmp.name
else:
    # 若本地有 best.pt 也允許使用（本地開發）
    candidate = os.path.join(os.path.dirname(__file__), "best.pt")
    if os.path.exists(candidate):
        weights_path = candidate

backend = None
model = None
if weights_path is None:
    st.sidebar.warning("尚未提供權重檔，請於此處上傳 .pt 後再進行推論。")
else:
    try:
        backend, model = load_model(weights_path)
        st.info(f"目前使用模型後端：{backend}\n\n權重來源：{('上傳檔案' if uploaded_w is not None else '本機 best.pt')}\n檔案：{weights_path}")
    except Exception as e:
        st.error(str(e))

# ======= 工具函式 =======

def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def predict_image(np_rgb: np.ndarray, backend: str, model: object) -> np.ndarray:
    """對單張 RGB 影像做推論並回傳已繪製標註的 RGB 影像。"""
    if backend == "ultralytics":
        # ultralytics: 直接指定 conf 參數，低於門檻者不會被繪製
        results = model.predict(source=np_rgb, conf=CONF_THRESH, verbose=False)
        # results[0].plot() 產生 BGR，需轉回 RGB
        bgr = results[0].plot()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    else:
        # yolov7 / yolov5: 於 model.conf 設門檻，並用 render() 取得繪製結果
        results = model(np_rgb)
        rendered = results.render()[0]  # BGR
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        return rgb


def predict_video_to_file(in_path: str, out_path: str, backend: str, model: object, progress_placeholder) -> Tuple[int, int, float]:
    """對影片逐幀推論並輸出 MP4 檔案，回傳 (width, height, fps)。"""
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

            # 轉成 RGB 以符合我們的 predict_image 輸入格式
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb = predict_image(frame_rgb, backend, model)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)

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
image_file = st.file_uploader("上傳圖片 (jpg/png/webp)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False, disabled=(model is None))

col1, col2 = st.columns(2)
with col1:
    if image_file is not None and model is not None and backend is not None:
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="原始圖片", use_column_width=True)
with col2:
    if image_file is not None and model is not None and backend is not None:
        with st.spinner("模型推論中…"):
            rgb = _pil_to_np_rgb(img)
            annotated = predict_image(rgb, backend, model)
            st.image(annotated, caption=f"推論結果（conf ≥ {CONF_THRESH:.2f}）", use_column_width=True)

st.markdown("---")
st.subheader("影片推論")
video_file = st.file_uploader("上傳影片 (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False, disabled=(model is None))

if video_file is not None and model is not None and backend is not None:
    # 將上傳影片存成暫存檔
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as src_tmp:
        src_tmp.write(video_file.read())
        src_path = src_tmp.name

    # 輸出檔名
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst_tmp:
        out_path = dst_tmp.name

    st.write("開始處理影片，請稍候（依影片長度而定）…")
    ph = st.empty()
    with st.spinner("影片推論中…"):
        w, h, fps = predict_video_to_file(src_path, out_path, backend, model, ph)

    # 顯示結果影片並提供下載
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

    # 清理暫存檔（Streamlit Cloud 為短暫環境，不清也可；這裡保守清理）
    try:
        os.remove(src_path)
    except Exception:
        pass
    # 保留 out_path，直到使用者下載完成或會話結束

st.markdown("---")
st.caption(
    "提示：優先使用 YOLOv7（torch.hub）載入 best.pt；只顯示信心度 ≥ 0.7 的偵測框。如模型未載入，請確認 best.pt 與 requirements.txt 相依關係是否已正確安裝。"
)
