# Jitter waveform guide

`jitter_waveform_guide.py` は、Matplotlib で **ideal waveform** と **jittered waveform** を比較表示するサンプルです。

## 何が見えるか

1. **Ideal waveform vs jittered waveform**
   - 理想的な NRZ 波形
   - ランダムジッタを加えた NRZ 波形
2. **Edge timing error**
   - 各エッジがどれだけ前後にずれたか
3. **Jitter histogram**
   - ジッタ量の分布
4. **Overlapped sine edges at one nominal timing**
   - 同じゼロクロス時刻を基準に複数回観測したサイン波を重ね描き
   - ジッタにより、1つのタイミング付近に複数のエッジが束になって見える

## 実行方法

```powershell
python rj/guides/jitter_waveform_guide.py
```

実行すると、同じフォルダに以下の PNG も保存されます。

```text
rj/guides/jitter_waveform_guide.png
```

## 主なパラメータ

`generate_jittered_waveform()` の引数で調整できます。

| 引数 | 意味 |
|------|------|
| `n_bits` | 波形の長さ（UI 数） |
| `ui` | 1 UI の長さ |
| `samples_per_ui` | 1 UI あたりの描画サンプル数 |
| `jitter_sigma_ui` | ランダムジッタの RMS 値（UI 単位） |
| `seed` | 再現性のための乱数シード |
| `jitter_clip_ui` | ジッタの最大振れ幅 |

## 使いどころ

- 「ジッタがあると波形の立ち上がり位置がどうずれるか」を見たいとき
- RJ のイメージを直感的に説明したいとき
- `Fail Count` や `BER` の前段にある波形イメージを掴みたいとき
- オシロスコープでエッジが太く見える理由を説明したいとき
