# Fail Count を微分するとヒストグラムになる理由

> このファイルは `rj/diff_failcount_to_histogram.md` から移動しました。

---

## 要約

Fail Count カーブ（補完 CDF）を微分すると、符号を反転した PDF（ヒストグラム）になります。

```
-d/dx Fail Count(x)  =  PDF(x)  =  ヒストグラム
```

- Fail Count が急峻な部分 → 傾きが大きい → その x にジッタが密集
- Fail Count が平坦な部分 → 傾きが小さい → ジッタがほぼない

## 数式の導出

```
Fail Count(x) = max_fail x (1 - CDF(x))

d/dx Fail Count(x) = -PDF(x)

したがって：
-d/dx Fail Count(x) = PDF(x)
```

## コードとの対応

```python
failcount_float = 100.0 * 0.5 * erfc((x - center) / (sigma * sqrt(2)))
pdf_numerical = -np.gradient(failcount_float, x)
```

詳細は `ber_analysis_guide.md` の「ヒストグラムの作り方」を参照。

