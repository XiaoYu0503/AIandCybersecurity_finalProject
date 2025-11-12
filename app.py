import io
import os
import tempfile
from typing import Tuple, List, Any

import numpy as np
import streamlit as st
from PIL import Image
import cv2
import requests
import importlib

try:
    import onnxruntime as ort
except Exception:
    ort = None  # 讓 UI 顯示提示

st.set_page_config(page_title="平交道障礙物即時辨識 (ONNX)", layout="wide")
st.title("平交道障礙物辨識 (Streamlit / ONNX)")
st.caption("預設採用 ONNXRuntime（.onnx）；也支援直接載入 YOLOv7 的 .pt（僅建議本地環境，雲端可能因套件相容性失敗）。")

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

def _get_secret(name: str) -> str | None:
    # 先讀 Streamlit secrets，再讀環境變數
    try:
        val = st.secrets.get(name)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name)

def download_to_temp(url: str, suffix: str) -> str:
    """從 URL 下載到臨時檔，盡量顯示進度。回傳檔案路徑。"""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get('Content-Length', 0))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    bytes_read = 0
    prog = st.progress(0) if total > 0 else None
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            tmp.write(chunk)
            bytes_read += len(chunk)
            if prog and total:
                prog.progress(min(bytes_read / total, 1.0))
    tmp.flush()
    if prog:
        prog.empty()
    return tmp.name

def download_from_gdrive(id_or_url: str, suffix: str) -> str:
    """使用 gdown 從 Google Drive 下載到臨時檔，支援分享連結或檔案 ID。"""
    try:
        gdown = importlib.import_module("gdown")
    except Exception as e:
        raise RuntimeError("需要安裝 gdown 才能從 Google Drive 抓取：pip install gdown") from e
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    # 若是純 ID，使用 id= 參數；若是完整網址，使用 url 參數
    is_url = id_or_url.strip().lower().startswith(("http://", "https://"))
    if is_url:
        ok = gdown.download(id_or_url, tmp.name, quiet=True, fuzzy=True)
    else:
        ok = gdown.download(id=id_or_url.strip(), output=tmp.name, quiet=True)
    if not ok:
        raise RuntimeError("gdown 下載失敗，請確認連結/ID 是否可公開存取或權限設定正確。")
    return tmp.name

def _is_probably_html(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            head = f.read(1024)
        head_lower = head.strip().lower()
        return head_lower.startswith(b'<!doctype') or head_lower.startswith(b'<html')
    except Exception:
        return False

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

def load_yolov7_pt(weights_path: str):
    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as e:
        raise RuntimeError("需要本地已安裝 PyTorch 才能載入 .pt：pip install torch torchvision") from e
    try:
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', path=weights_path, trust_repo=True, force_reload=False)
        try:
            model.to('cpu')
            model.eval()
            model.conf = CONF_THRESH
        except Exception:
            pass
        return model
    except Exception as e:
        raise RuntimeError(f"YOLOv7 .pt 載入失敗：{e}") from e

# ======= 模型權重選擇（側邊欄） =======
st.sidebar.header("模型設定")
# 根據 secrets MODEL_FORMAT 預設後端（onnx/pt）
fmt_secret = (_get_secret("MODEL_FORMAT") or "onnx").lower()
default_idx = 0 if fmt_secret.startswith("onnx") else 1
model_type = st.sidebar.radio("選擇權重格式", ["ONNX (.onnx)", "YOLOv7 PyTorch (.pt)"], index=default_idx)
help_txt = "ONNX：建議用於雲端；.pt：僅建議本地且需已安裝 torch"
uploaded_w = st.sidebar.file_uploader("上傳權重檔", type=["onnx", "pt"], help=help_txt)
url_w = st.sidebar.text_input("或輸入權重 URL", placeholder="https://.../model.onnx 或 best.pt")
gd_w = st.sidebar.text_input("或輸入 Google Drive 連結/ID")
btn_gd_w = st.sidebar.button("從 Google Drive 抓取權重")

backend_name: str | None = None
backend_obj: object | None = None

if uploaded_w is not None:
    suffix = ".onnx" if uploaded_w.name.lower().endswith(".onnx") else ".pt"
    wtmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    wtmp.write(uploaded_w.read())
    wtmp.flush()
    weights_path = wtmp.name
else:
    weights_path = None
    # 本地預設：存在 model.onnx 則自動載入
    if model_type.startswith("ONNX"):
        candidate = os.path.join(os.path.dirname(__file__), "model.onnx")
        if os.path.exists(candidate):
            weights_path = candidate
    # 1) 若 secrets 設定了 MODEL_URL，優先自動下載
    if weights_path is None:
        model_url_secret = _get_secret("MODEL_URL")
        if model_url_secret:
            lower = model_url_secret.lower()
            suffix = ".onnx" if (fmt_secret.startswith("onnx") or lower.endswith(".onnx")) else ".pt"
            with st.spinner("從 MODEL_URL 自動下載權重中…"):
                try:
                    weights_path = download_to_temp(model_url_secret, suffix)
                    st.sidebar.success("已從 MODEL_URL 自動下載權重")
                except Exception as e:
                    st.sidebar.error(f"MODEL_URL 下載失敗：{e}")
                    weights_path = None
    # 2) 若 secrets 設定了 MODEL_GDRIVE_ID 或 MODEL_GDRIVE_URL，自動 gdown 下載
    if weights_path is None:
        gd_secret = _get_secret("MODEL_GDRIVE_ID") or _get_secret("MODEL_GDRIVE_URL")
        if gd_secret:
            suffix = ".onnx" if fmt_secret.startswith("onnx") else ".pt"
            with st.spinner("從 Google Drive (secrets) 自動下載權重中…"):
                try:
                    weights_path = download_from_gdrive(gd_secret, suffix)
                    st.sidebar.success("已從 Google Drive 自動下載權重")
                except Exception as e:
                    st.sidebar.error(f"Google Drive 下載失敗：{e}")
                    weights_path = None
    # 如果提供 URL，優先用 URL 下載
    if weights_path is None and url_w:
        # 依 model_type 或 URL 副檔名決定後綴
        lower = url_w.lower()
        suffix = ".onnx" if (model_type.startswith("ONNX") or lower.endswith(".onnx")) else ".pt"
        with st.spinner("下載權重中…"):
            try:
                weights_path = download_to_temp(url_w, suffix)
            except Exception as e:
                st.error(f"下載權重失敗：{e}")
                weights_path = None
    # 若提供 Google Drive 連結/ID 且按下按鈕，使用 gdown 抓取
    if weights_path is None and gd_w and btn_gd_w:
        suffix = ".onnx" if model_type.startswith("ONNX") else ".pt"
        with st.spinner("從 Google Drive 下載權重中…"):
            try:
                weights_path = download_from_gdrive(gd_w, suffix)
            except Exception as e:
                st.error(f"Google Drive 下載失敗：{e}")
                weights_path = None

# 載入對應 backend
if weights_path is None:
    st.sidebar.warning("尚未提供權重檔，請先上傳。ONNX 最穩定；.pt 僅建議本地。")
    _assumption_notice()
else:
    lower = weights_path.lower()
    is_onnx = lower.endswith(".onnx")
    is_pt = lower.endswith(".pt")
    if is_onnx:
        if not model_type.startswith("ONNX"):
            st.sidebar.info("已自動依檔案副檔名改用 ONNX 後端。")
        if ort is None:
            st.error("onnxruntime 未安裝，請確認 requirements.txt 已包含 onnxruntime。")
        else:
            try:
                backend_obj = load_session(weights_path)
                backend_name = "onnx"
                st.success(f"已載入 ONNX 模型：{os.path.basename(weights_path)}")
            except Exception as e:
                # 針對常見情境提供更明確提示
                extra = ""
                try:
                    sz = os.path.getsize(weights_path)
                    if sz < 200 * 1024:
                        extra += "\n檔案大小異常偏小，可能是未授權或下載到 HTML/錯誤頁面。"
                    if _is_probably_html(weights_path):
                        extra += "\n偵測到下載內容像是 HTML 頁面：請確認 Google Drive 檔案已設為「擁有連結者可查看」，或改用檔案 ID/直接下載連結。"
                except Exception:
                    pass
                if "EfficientNMS_TRT" in str(e):
                    extra += "\n此 ONNX 內含 TensorRT 的 EfficientNMS_TRT 節點，請重新匯出為非 end2end 的 CPU 版 ONNX（不要加 --end2end）。"
                if "INVALID_PROTOBUF" in str(e) or "Protobuf" in str(e):
                    extra += "\nProtobuf 解析失敗：通常是下載到的不是 .onnx（例如 Google 驗證頁），或檔案已損毀/未完整。"
                st.error(f"ONNX 載入失敗：{e}{extra}")
                _assumption_notice()
    elif is_pt:
        if not model_type.startswith("YOLOv7"):
            st.sidebar.info("已自動依檔案副檔名改用 YOLOv7 (.pt) 後端。")
        try:
            backend_obj = load_yolov7_pt(weights_path)
            backend_name = "yolov7"
            st.info("偵測使用 YOLOv7 .pt（僅建議本地）。")
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        st.error("無法判斷權重檔格式，請提供 .onnx 或 .pt。")
        st.stop()

def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def predict_image(np_rgb: np.ndarray, backend: str, obj: object) -> np.ndarray:
    if backend == "onnx":
        inp, scale, left, top = preprocess(np_rgb)
        inputs = {obj.get_inputs()[0].name: inp}
        outputs = obj.run(None, inputs)
        annotated = postprocess(outputs, np_rgb, scale, left, top)
        return annotated
    elif backend == "yolov7":
        results = obj(np_rgb)  # YOLOv7 results
        rendered = results.render()[0]  # BGR
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        return rgb
    else:
        raise RuntimeError("未知的 backend")

def predict_video_to_file(in_path: str, out_path: str, backend: str, obj: object, progress_placeholder) -> Tuple[int, int, float]:
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
            annotated_rgb = predict_image(frame_rgb, backend, obj)
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
image_file = st.file_uploader("上傳圖片 (jpg/png/webp)", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False, disabled=(backend_obj is None))
with st.expander("或輸入圖片 URL/Google Drive 推論"):
    url_img = st.text_input("圖片 URL", placeholder="https://.../image.jpg")
    gd_img = st.text_input("圖片 Google Drive 連結/ID")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_img = st.button("從 URL 下載並推論", disabled=(backend_obj is None))
    with col_btn2:
        run_img_gd = st.button("從 Google Drive 下載並推論", disabled=(backend_obj is None))

col1, col2 = st.columns(2)
with col1:
    if image_file is not None and backend_obj is not None and backend_name is not None:
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="原始圖片", use_column_width=True)
    elif run_img and url_img and backend_obj is not None and backend_name is not None:
        with st.spinner("下載圖片中…"):
            try:
                img_path = download_to_temp(url_img, suffix=".jpg")
                img = Image.open(img_path)
                st.image(img, caption="原始圖片 (URL)", use_column_width=True)
            except Exception as e:
                st.error(f"下載圖片失敗：{e}")
    elif run_img_gd and gd_img and backend_obj is not None and backend_name is not None:
        with st.spinner("從 Google Drive 下載圖片中…"):
            try:
                img_path = download_from_gdrive(gd_img, suffix=".jpg")
                img = Image.open(img_path)
                st.image(img, caption="原始圖片 (Google Drive)", use_column_width=True)
            except Exception as e:
                st.error(f"Google Drive 圖片下載失敗：{e}")
with col2:
    if image_file is not None and backend_obj is not None and backend_name is not None:
        with st.spinner("模型推論中…"):
            rgb = _pil_to_np_rgb(img)
            annotated = predict_image(rgb, backend_name, backend_obj)
            st.image(annotated, caption=f"推論結果（conf ≥ {CONF_THRESH:.2f}）", use_column_width=True)
    elif run_img and url_img and backend_obj is not None and backend_name is not None:
        with st.spinner("模型推論中…"):
            rgb = _pil_to_np_rgb(img)
            annotated = predict_image(rgb, backend_name, backend_obj)
            st.image(annotated, caption=f"推論結果（URL）（conf ≥ {CONF_THRESH:.2f}）", use_column_width=True)

st.markdown("---")
st.subheader("影片推論")
video_file = st.file_uploader("上傳影片 (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False, disabled=(backend_obj is None))
with st.expander("或輸入影片 URL/Google Drive 推論"):
    url_vid = st.text_input("影片 URL", placeholder="https://.../video.mp4")
    gd_vid = st.text_input("影片 Google Drive 連結/ID")
    col_btnv1, col_btnv2 = st.columns(2)
    with col_btnv1:
        run_vid = st.button("從 URL 下載並推論影片", disabled=(backend_obj is None))
    with col_btnv2:
        run_vid_gd = st.button("從 Google Drive 下載並推論影片", disabled=(backend_obj is None))

if video_file is not None and backend_obj is not None and backend_name is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as src_tmp:
        src_tmp.write(video_file.read())
        src_path = src_tmp.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst_tmp:
        out_path = dst_tmp.name
    st.write("開始處理影片，請稍候（依影片長度而定）…")
    ph = st.empty()
    with st.spinner("影片推論中…"):
        w, h, fps = predict_video_to_file(src_path, out_path, backend_name, backend_obj, ph)
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

elif run_vid and url_vid and backend_obj is not None and backend_name is not None:
    # 從 URL 下載影片並推論
    with st.spinner("下載影片中…"):
        try:
            src_path = download_to_temp(url_vid, suffix=os.path.splitext(url_vid)[1] or ".mp4")
        except Exception as e:
            st.error(f"下載影片失敗：{e}")
            src_path = None
    if src_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst_tmp:
            out_path = dst_tmp.name
        st.write("開始處理影片，請稍候（依影片長度而定）…")
        ph = st.empty()
        with st.spinner("影片推論中…"):
            w, h, fps = predict_video_to_file(src_path, out_path, backend_name, backend_obj, ph)
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
elif run_vid_gd and gd_vid and backend_obj is not None and backend_name is not None:
    with st.spinner("從 Google Drive 下載影片中…"):
        try:
            src_path = download_from_gdrive(gd_vid, suffix=".mp4")
        except Exception as e:
            st.error(f"Google Drive 影片下載失敗：{e}")
            src_path = None
    if src_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as dst_tmp:
            out_path = dst_tmp.name
        st.write("開始處理影片，請稍候（依影片長度而定）…")
        ph = st.empty()
        with st.spinner("影片推論中…"):
            w, h, fps = predict_video_to_file(src_path, out_path, backend_name, backend_obj, ph)
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

