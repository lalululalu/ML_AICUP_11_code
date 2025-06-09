# ML_AICUP_11_code
---
# AI CUP 2025 - 桌球擊球資訊預測專案
```
本專案為參加 AI CUP 2025 桌球擊球資料預測競賽而建立，目標是根據感測器數據預測擊球者的個人屬性：`gender`（性別）、`hold`（持拍手）、`play years`（打球年資）、`level`（程度）。
```
## 專案結構

```

├── aicup.ipynb           # 主程式（特徵萃取、訓練、預測、評估）
├── data/
│   ├── 39\_Training\_Dataset/
│   └── 39\_Test\_Dataset/
└── output/
└── submission.csv    # 預測結果（機率格式）

````

## 環境建置

### 1. Python 版本建議

```bash
Python 3.9 以上
````

### 2. 套件安裝

請使用 pip 安裝相依套件：

```bash
pip install -r requirements.txt
```

**requirements.txt**

```txt
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
tqdm
joblib
```

> 若使用 GPU 或延伸深度學習模型，請自行配置對應的 CUDA 驅動。

## 輸入資料格式

資料夾需置於以下位置（預設路徑）：

```
/content/drive/MyDrive/我的電腦/我的MacBook Pro/AI cup/
├── 39_Training_Dataset/
└── 39_Test_Dataset/
```

每筆檔案包含一筆擊球的時間序列資料，資料來自 IMU 感測器。資料需先經過 `FFT`（快速傅立葉轉換）萃取頻域特徵再進行建模。

## 輸出資料格式

最終預測結果為 `submission.csv`，內容格式如下（每一欄為對應類別的機率）：

```csv
uuid,gender_f,gender_m,hold_left,hold_right,play_years_0,...,level_4
0000,0.01,0.99,0.8,0.2,0.1,...,0.9
```

## 核心模組說明

* `extract_features_fft(df)`：將時間序列轉換為頻域特徵。
* `train_random_forest(X, y)`：訓練 `Random Forest` 分類器。
* `predict_proba(models, X_test)`：輸出各分類機率。
* `evaluate_model(y_true, y_pred)`：以 ROC AUC（含 binary + micro-average）評估模型表現。

## 重現流程

1. 開啟 Google Colab 並掛載 Google Drive。
2. 確認資料集置於指定路徑。
3. 依序執行 `aicup.ipynb`：

   * 對訓練與測試資料做 FFT 特徵轉換。
   * 訓練 `Random Forest` 模型。
   * 預測並輸出 `submission.csv`。

## 聯絡方式

若對本專案有任何問題或建議，歡迎聯絡隊長：
**張庭瑋（Ting-Wei Chang）** — ` tony51104@gmail.com`

---
