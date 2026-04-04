# Signal Integrity Notes

シグナルインテグリティに関する解説資料・ツール集です。

## フォルダ構成

```
signal-integrity-notes/
├── requirements.txt
├── rj/                          # Random Jitter 解析パッケージ
│   ├── app.py                   # Streamlit アプリ
│   ├── rj_app_core.py           # アプリコアロジック
│   ├── ber.py                   # BER / RJ コア計算
│   ├── smoke_test.py            # スモークテスト
│   ├── README.md                # RJ アプリの詳細 README
│   └── guides/                  # 解説資料
│       ├── ber_analysis_guide.md
│       ├── ber_terminology.md
│       ├── diff_failcount_to_histogram.md
│       ├── cdf_equals_failcount.py
│       ├── integration_differentiation_relation.py
│       ├── why_derivative_to_histogram.py
│       ├── jitter_waveform_guide.py
│       ├── jitter_waveform_guide.md
│       ├── sine_edges_to_failcount_guide.py
│       └── sine_edges_to_failcount_guide.md
├── reflection/                  # 反射（Reflection）解説
│   └── reflection_guide.py
└── spara/                       # S パラメータ解説
    └── spara_guide.py
```

## 各トピック

### Random Jitter (RJ) 解析 → `rj/`

Fail Count データから Random Jitter を抽出する Streamlit アプリ。

```powershell
# インストール
pip install -r requirements.txt

# アプリ起動
streamlit run rj/app.py

# スモークテスト
python -m rj.smoke_test

# ジッタ付き波形デモ（NRZ + overlapped sine edges）
python rj/guides/jitter_waveform_guide.py

# overlapped sine edges -> Fail Count -> 微分ヒストグラム
python rj/guides/sine_edges_to_failcount_guide.py
```

### 反射（Reflection）解説 → `reflection/`

伝送線路の反射係数・リターンロス・TDR・定在波（VSWR）の解説グラフを生成します。

```powershell
python reflection/reflection_guide.py
```

### S パラメータ解説 → `spara/`

2 ポートネットワークの S パラメータ（S11・S21・位相・グループ遅延）の解説グラフを生成します。

```powershell
python spara/spara_guide.py
```

## インストール

```powershell
pip install -r requirements.txt
```

