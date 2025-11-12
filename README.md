# 平交道障礙物辨識（Streamlit App）

以本地 `best.pt`（YOLOv7 權重為主）進行圖片與影片偵測，信心度低於 0.7 的框不顯示。可部署至 Streamlit Cloud。

## 功能
- 上傳圖片（jpg/png/webp）並即時顯示推論結果。
- 上傳影片（mp4/mov/avi/mkv），逐幀推論並輸出繪製後影片。
- 只顯示信心度 `≥ 0.7` 的偵測框。

## 專案結構
```
.
├─ app.py              # Streamlit 主程式
├─ best.pt             # 訓練完成的模型權重（需自行提供；檔案大請勿直接 push）
├─ requirements.txt    # 相依套件清單
└─ README.md
```

## 本地執行方式
1. 於專案根目錄（與 `app.py` 同層）放置 `best.pt`。
2. 建議使用虛擬環境並安裝相依：
   ```bash
   pip install -r requirements.txt
   ```
3. 啟動：
   ```bash
   streamlit run app.py
   ```

## 部署到 Streamlit Cloud
1. 將此資料夾推送到 GitHub（例如：`XiaoYu0503/AIandCybersecurity_finalProject`），注意不要把超過 100MB 的 `best.pt` 推到 GitHub（已在 .gitignore 排除）。
2. 在 Streamlit Cloud 建立新 App：
   - Repository：選擇上述 Repo
   - Branch：main
   - Main file path：`app.py`
3. 啟動後，於 UI 上傳圖片或影片即可推論。

> 注意：
> - 優先以 torch.hub 載入 YOLOv7（需要可連網以抓取 `WongKinYiu/yolov7` hub），失敗時再嘗試 ultralytics 與 YOLOv5 後備。
> - 部署於 Cloud 時，由於 GitHub 有 100MB 限制，建議：
>   1) 在 Streamlit Cloud 的儲存空間手動上傳 `best.pt`（與 `app.py` 同層），或
>   2) 於啟動時從您可存取的雲端（例如 S3、GDrive 直鏈）下載 `best.pt` 至臨時目錄再載入。
> - 本專案偏向 CPU 推論（未強制指定 GPU），推論速度取決於影片長度與模型大小。
> - 若 Cloud 環境的 OpenCV 不支援某些編碼器，影片寫檔可能失敗；可改傳短片、降低解析度或改用其他容器（如 .avi）。

## 授權
僅示範用途，依您資料與模型而定。
