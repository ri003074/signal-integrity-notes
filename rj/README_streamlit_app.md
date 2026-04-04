# Fail Count to RJ Streamlit App

This app lets you input Fail Count data and visualize:

1. Input Fail Count (and BER)
2. Numerical derivative (`-d(Fail Count)/dx`)
3. Q-transform with linear fit for RJ extraction
4. RMS-based RJ estimate from derivative-derived density (`rms_sigma`, `rms_mu`)

## Files

- `rj/failcount_rj_streamlit_app.py`: Streamlit UI
- `rj/rj_app_core.py`: parsing/validation/analysis logic
- `rj/smoke_test_failcount_app.py`: quick logic smoke test

## Install

```powershell
pip install -r requirements.txt
```

## Run Smoke Test

```powershell
python -m rj.smoke_test_failcount_app
```

## Run Streamlit App

```powershell
streamlit run rj/failcount_rj_streamlit_app.py
```

## Supported Input

- Sample generator
- Paste values (comma/space/newline)
- CSV upload
  - `x,failcount` (header)
  - one column (`failcount` only)
  - two columns without header (`x,failcount`)
