# 平交道障礙物辨識（Streamlit App, ONNX）

改用 ONNXRuntime（CPU）做推論，避免在 Streamlit Cloud（Python 3.13）上安裝 torch/torchvision 造成的失敗；信心度低於 0.7 的框不顯示。

## 功能
- 上傳圖片（jpg/png/webp）並即時顯示推論結果。
- 上傳影片（mp4/mov/avi/mkv），逐幀推論並輸出繪製後影片。
- 只顯示信心度 `≥ 0.7` 的偵測框。
- 側邊欄上傳 ONNX 權重（.onnx），未上傳前不會載入模型（避免 Cloud 缺檔）。

## 專案結構
```
.
├─ app.py              # Streamlit 主程式
├─ model.onnx          # (可選) 本地開發用之 ONNX 權重檔（部署時建議改由 UI 上傳）
├─ requirements.txt    # 相依套件清單
├─ runtime.txt         # (可選) 舊式 Python 版本宣告，Cloud 可能忽略
├─ packages.txt        # 系統層必要套件 (libgl, ffmpeg 等)
└─ README.md
```

## 本地執行方式
1. 建議使用虛擬環境並安裝相依：
   ```bash
   pip install -r requirements.txt
   ```
2. 啟動：
   ```bash
   streamlit run app.py
   ```
3. 於側邊欄上傳 `model.onnx`（或放置在專案根目錄使用預設 `model.onnx`）。

## 部署到 Streamlit Cloud
1. 將此資料夾推送到 GitHub（例如：`XiaoYu0503/AIandCybersecurity_finalProject`），避免把大型權重檔推到 GitHub（.gitignore 已排除 .pt）。
2. 在 Streamlit Cloud 建立新 App：
   - Repository：選擇上述 Repo
   - Branch：main
   - Main file path：`app.py`
3. 啟動後可以：
   - 直接使用自動下載的權重（建議用 ONNX）。
   - 或於 UI 上傳圖片/影片、或貼上 URL/Google Drive 連結一鍵抓取推論。

> 注意：
> - 本版本以 ONNXRuntime (CPU) 推論為預設（雲端穩定），也支援選用 YOLOv7 .pt（本地建議）。
> - 若 `cv2` 載入失敗，請確認 `packages.txt` 已含 `libgl1`、`libglib2.0-0`、`ffmpeg`，且使用 headless 版本。
> - OpenCV 編碼器在雲端環境可能受限；若影片寫檔失敗，請嘗試較短影片或不同容器。
> - 若您只有 `.pt` 權重，建議先在本地轉為 `model.onnx`；若要在雲端直接使用 .pt，請自行確保 torch 能在該環境成功安裝。

### 自動下載模型（推薦雲端設定）

你可以在 Streamlit Cloud 設定 Secrets 或環境變數，App 啟動時會自動下載權重：

- MODEL_FORMAT: onnx 或 pt（預設 onnx，會影響預設後端與下載檔案副檔名）
- MODEL_URL: 權重的直接下載 URL（http/https）
- MODEL_GDRIVE_ID 或 MODEL_GDRIVE_URL: Google Drive 檔案 ID 或分享連結（由 gdown 下載）

範例（Secrets）:
```
MODEL_FORMAT = "onnx"
MODEL_GDRIVE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"  # 你的權重檔案 ID
```
App 啟動時會自動下載並載入權重；同時你也可以在 UI 內手動輸入 URL/Drive 連結，或上傳本地檔案。

## 授權
僅示範用途，依您資料與模型而定。
