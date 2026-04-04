# Fail Count to RJ Streamlit App

This app lets you input Fail Count data and visualize:

1. Input Fail Count (and BER)
2. Numerical derivative (`-d(Fail Count)/dx`)
3. Q-transform with linear fit for RJ extraction
4. RMS-based RJ estimate from derivative-derived density (`rms_sigma`, `rms_mu`)

## Files

### App

| ファイル | 説明 |
|---------|------|
| `rj/app.py` | Streamlit UI |
| `rj/rj_app_core.py` | parsing / validation / analysis logic |
| `rj/ber.py` | BER / RJ core math (Q function, extract_rj) |
| `rj/smoke_test.py` | quick logic smoke test |

### Guides（説明資料）

| ファイル | 説明 |
|---------|------|
| `rj/guides/ber_analysis_guide.md` | BER / RJ 解析ガイド |
| `rj/guides/ber_terminology.md` | `ber.py` に出てくる用語集 |
| `rj/guides/diff_failcount_to_histogram.md` | Fail Count 微分 → ヒストグラムの解説 |
| `rj/guides/cdf_equals_failcount.py` | CDF = Fail Count の図解スクリプト |
| `rj/guides/integration_differentiation_relation.py` | 積分・微分の関係図解スクリプト |
| `rj/guides/why_derivative_to_histogram.py` | 微分でヒストグラムになる理由の図解スクリプト |
| `rj/guides/jitter_waveform_guide.py` | Ideal / jittered waveform と overlapped sine edges の可視化スクリプト |
| `rj/guides/jitter_waveform_guide.md` | ジッタ波形ガイド |
| `rj/guides/sine_edges_to_failcount_guide.py` | Overlapped sine edges から Fail Count と histogram を作るガイド |
| `rj/guides/sine_edges_to_failcount_guide.md` | 上記ガイドの説明 |

## Install

```powershell
pip install -r requirements.txt
```

## Run Smoke Test

```powershell
python -m rj.smoke_test
```

## Run Streamlit App

```powershell
streamlit run rj/app.py
```

## Run Jitter Waveform Demo

```powershell
python rj/guides/jitter_waveform_guide.py
```

## Run Sine-edges to Fail Count Demo

```powershell
python rj/guides/sine_edges_to_failcount_guide.py
```

## Supported Input

- Sample generator
- Paste values (comma/space/newline)
- CSV upload
  - `x,failcount` (header)
  - one column (`failcount` only)
  - two columns without header (`x,failcount`)



