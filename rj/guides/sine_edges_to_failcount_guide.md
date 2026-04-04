# Sine edges to Fail Count guide

`sine_edges_to_failcount_guide.py` は、次の関係を 1 枚で見せるための図です。

```text
overlapped sine edges
    -> crossing-time spread
    -> Fail Count (complementary CDF)
    -> histogram density
```

## 何をしているか

同じ rising edge を何回も観測すると、ジッタによって zero-crossing time が少しずつ前後にずれます。

この crossing time の集まりに対して、時刻 `t` ごとに

```text
Fail Count(t) = crossing_time > t の本数
```

と定義すると、Fail Count は時刻とともに減少するカーブになります。

これは crossing time 分布の **補完 CDF** です。

この図では、Fail Count と crossing time のヒストグラムを並べて、
両者の対応関係が直感的に見えるようにしています。

## 図の見方

1. **Overlapped sine edges**
   - ジッタで複数のエッジが 1 つのタイミング付近に重なる
2. **Fail Count**
   - サンプリング時刻を右に動かすと、まだ crossing していない波形数が減っていく
3. **Histogram of crossing times**
   - crossing time がどこに多いかを示す

## 実行方法

```powershell
python rj/guides/sine_edges_to_failcount_guide.py
```

PNG も同じフォルダに保存されます。

```text
rj/guides/sine_edges_to_failcount_guide.png
```
