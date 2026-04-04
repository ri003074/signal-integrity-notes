# ber.py 用語解説

`rj/ber.py` に登場する数学・統計・信号処理用語の解説です。

---

## σ (sigma) ／ 標準偏差

分布の「幅」を表す指標。**σ が小さい = ジッタが小さい**。

| 変数名 | 意味 |
|-------|------|
| `sigma` | シミュレーション用の真値の σ |
| `sigma_rj` | Q フィットで推定した RJ の σ |
| `TRUE_SIGMA` | シミュレーションに設定した真の σ |
| `noise_sigma` | 加算する測定ノイズの σ |
| `rms_sigma` | RMS ベースで推定した σ（重み付き分散） |

抽出式: `σ_RJ = 1 / slope`（直線フィットの傾きの逆数）

---

## μ (mu) ／ 平均・中心

分布の中心位置。**Fail Count が 50% になる x の値**。

| 変数名 | 意味 |
|-------|------|
| `center` | 入力パラメータとしての中心位置 |
| `mu` | Q フィットで推定した中心位置 |
| `TRUE_CENTER` | シミュレーションに設定した真の中心 |
| `rms_mu` | RMS ベースで推定した中心（重み付き平均） |

抽出式: `μ = −intercept / slope`

---

## 正規分布（ガウス分布）

ランダムなばらつき（ノイズ・ジッタ）が従う最も基本的な確率分布。グラフはベル型。

```
f(x) = 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)^2 / (2*sigma^2))
```

パラメータ μ（中心）と σ（幅）だけで完全に形が決まる。

---

## PDF（確率密度関数）

**Probability Density Function** — ある x 付近の確率の密度。ヒストグラムの棒の高さ。

```
PDF(x) が高い → その x にデータが密集
PDF(x) が低い → その x にデータがほとんどない
```

Fail Count を微分すると PDF が得られる:

```
PDF(x) = -d(Fail Count)/dx = -np.gradient(failcount, x)
```

---

## CDF（累積分布関数）

**Cumulative Distribution Function** — 「x 以下になる確率」。PDF を積分したもの。

```
CDF(x) = P(X <= x) = integral_{-inf}^{x} PDF(t) dt
```

0 から 1 に単調増加する S 字カーブ。

---

## 補完 CDF（CCDF）

**Complementary CDF** — 「x より大きくなる確率」。CDF を上下反転したもの。

```
CCDF(x) = 1 - CDF(x) = P(X > x)
```

**Fail Count カーブの正体がこれ:**

```python
failcount = max_fail * 0.5 * erfc((x - center) / (sigma * np.sqrt(2)))
```

---

## erf（誤差関数）

**Error function**

```
erf(x) = (2/sqrt(pi)) * integral_0^x exp(-t^2) dt
```

- `erf(0) = 0`、`erf(inf) = 1`、`erf(-inf) = -1`

正規分布の CDF を計算するための基本関数。

---

## erfc（補完誤差関数）

**Complementary error function**

```
erfc(x) = 1 - erf(x) = (2/sqrt(pi)) * integral_x^inf exp(-t^2) dt
```

- `erfc(0) = 1`、`erfc(inf) = 0`

正規分布の CCDF との関係:

```
Q(x) = CCDF(x) = 0.5 * erfc(x / sqrt(2))
```

使用箇所:

```python
from scipy.special import erfc
failcount = max_fail * 0.5 * erfc((x - center) / (sigma * np.sqrt(2)))
```

---

## erfinv（逆誤差関数）

**Inverse error function** — erf の逆関数。`erf(y) = x` のとき `erfinv(x) = y`。

Q 逆関数との関係: `Q_inv(p) = sqrt(2) * erfinv(1 - 2*p)`

`ber.py` では直接使わず、同等の計算を `ndtri` で行っています。

---

## Q 関数

標準正規分布が x を超える確率。

```
Q(x) = P(X > x),  X ~ N(0,1)
     = 0.5 * erfc(x / sqrt(2))
```

代表値:

| x | Q(x) |
|---|------|
| 0 | 0.500 |
| 1 | 0.159 |
| 2 | 0.023 |
| 3 | 0.00135 |

BER との関係: `BER(x) = Q((x - mu) / sigma_RJ)`

```python
def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))
```

---

## Q 逆関数（q_inv）

BER の値から対応する正規化値を返す逆関数。

```
Q_inv(p) = -ndtri(p) = sqrt(2) * erfinv(1 - 2*p)
```

**RJ 抽出の核心 — BER の式に Q 逆関数を適用すると直線になる:**

```
BER(x) = Q((x - mu) / sigma_RJ)
  → 両辺に Q 逆関数を適用
Q_inv(BER) = (1/sigma_RJ)*x - mu/sigma_RJ   ← 直線の式!
```

```python
def q_inv(ber):
    ber = np.clip(ber, 1e-15, 1 - 1e-15)
    return -ndtri(ber)
```

---

## ndtri（正規分布の逆 CDF）

**Inverse of the normal CDF (percent point function)**

「P(X <= x) = p となる x」を返す。

```
ndtri(p) = x  such that  Phi(x) = p
Q_inv(p) = -ndtri(p)
```

```python
from scipy.special import ndtri
```

---

## BER（ビットエラーレート）

**Bit Error Rate** — テスト失敗の割合（0 〜 1 に正規化）。

```
BER = Fail Count / max_fail
```

| BER | 意味 |
|-----|------|
| 1.0 | 全試行が失敗（最悪） |
| 0.5 | 半分が失敗（遷移中心） |
| 0.0 | 全試行が成功（最良） |

---

## Fail Count

テストの失敗回数。横軸 x を変化させながら測定した値。

```
x が小さい（マージンが少ない） → Fail Count 大
x が大きい（マージンが多い）   → Fail Count 小
```

形状は正規分布の補完 CDF（CCDF）:

```
Fail Count(x) = max_fail * Q((x - mu) / sigma_RJ)
```

---

## max_fail

Fail Count の最大値（= 総試行回数）。BER の正規化に使用。

```
BER = Fail Count / max_fail
```

一般的な値: 100、1000、10000 など。

---

## ber_range

Q 逆関数変換の有効範囲（下限・上限の BER 値のタプル）。デフォルト: `(0.05, 0.95)`

BER が 0 または 1 に近いと Q 逆関数が ±∞ に発散するため、中間領域だけでフィットする:

```python
mask = (ber > ber_range[0]) & (ber < ber_range[1])
```

---

## RJ（Random Jitter）

**Random Jitter** — 正規分布に従うランダムなジッタ成分。

- 発生源: 熱雑音、電源ノイズ、位相雑音
- 特徴: 理論上、裾野が無限に続く（有界でない）
- パラメータ: σ_RJ と μ で完全に記述できる
- Q-scale プロットが直線になる

---

## DJ（Deterministic Jitter）

**Deterministic Jitter** — 規則性・再現性のあるジッタ成分（`ber.py` では直接扱わないが参考）。

- 発生源: クロストーク、ISI（符号間干渉）、電源変動
- 特徴: 有界（上限がある）
- Q-scale プロットが直線からずれる場合に DJ 混在を疑う

---

## linregress（線形回帰）

最小二乗法で直線 `y = slope*x + intercept` をフィット。

```python
from scipy.stats import linregress
result = linregress(x_fit, q_inv_fit)
slope     = result.slope
intercept = result.intercept
r_squared = result.rvalue ** 2
```

---

## slope（傾き）

直線の傾き。RJ 抽出では `slope = 1 / sigma_RJ`。

```
sigma_RJ = 1 / slope
```

傾きが急 → sigma_RJ が小さい → ジッタが小さい。

---

## intercept（切片）

直線の y 切片（x=0 のときの y 値）。RJ 抽出では `intercept = -mu / sigma_RJ`。

```
mu = -intercept / slope
```

---

## R²（決定係数）

**R-squared / Coefficient of determination** — 回帰直線の当てはまりの良さ（0 〜 1）。

| R² | 意味 |
|----|------|
| 1.0 | 完全にフィット（全点が直線上） |
| > 0.99 | 非常に良好 |
| < 0.95 | DJ 混在の疑いあり |

```python
r_squared = result.rvalue ** 2
```

---

## np.gradient（数値微分）

隣り合う点の差分から傾きを計算（中心差分法）。

```python
pdf_numerical = -np.gradient(failcount_float, x)
# => -d(Fail Count)/dx = PDF
```

内部点: `gradient[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])`

---

## np.clip

配列の値を指定範囲に制限する。

```python
np.clip(pdf_raw, 0, None)        # 負値を 0 にクリップ
np.clip(ber, 1e-15, 1 - 1e-15)  # 0 と 1 を避ける（Q 逆関数の発散防止）
```

---

## seed（乱数シード）

乱数生成の初期値。同じ seed を指定すると毎回同じ結果（再現性の確保）。

```python
rng = np.random.default_rng(seed=7)
```

---

## 用語の相互関係まとめ

```
【入力】
  Fail Count(x)  ← 測定値（補完 CDF の形）
  max_fail       ← 総試行回数

【正規化】
  BER = Fail Count / max_fail

【Q 逆関数変換】
  Q_inv(BER) = -ndtri(BER)

【線形回帰（ber_range 内の有効域のみ）】
  Q_inv(BER) = slope * x + intercept

【パラメータ抽出】
  sigma_RJ = 1 / slope
  mu       = -intercept / slope
  R²       = rvalue²
```

### 関数名の対応表

| 用語 | Python | 数式 |
|-----|--------|------|
| 誤差関数 | `scipy.special.erf` | `(2/sqrt(pi)) int_0^x e^{-t^2} dt` |
| 補完誤差関数 | `scipy.special.erfc` | `1 - erf(x)` |
| 逆誤差関数 | `scipy.special.erfinv` | `erf^{-1}(x)` |
| Q 関数 | `q_function(x)` | `0.5 * erfc(x/sqrt(2))` |
| Q の逆関数 | `q_inv(ber)` | `-ndtri(ber)` |
| 正規分布の逆 CDF | `scipy.special.ndtri` | `Phi^{-1}(p)` |
| 正規分布 PDF | `scipy.stats.norm.pdf` | `1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))` |
| 数値微分 | `numpy.gradient` | `df/dx ≈ Δf/Δx` |

