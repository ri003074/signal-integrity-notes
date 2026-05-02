"""
バースト Sin 波形と反射の影響デモ

「バーストの前半は反射の影響ゼロ、後半から反射波が重なって波形が乱れる」
という現象を3段パネルで視覚的に示す。

  ① 入射波 V_inc(t)
  ② 反射波 V_ref(t) = Γ × V_inc(t − 2TD)  ← 2TD 遅れて到達
  ③ 合成波 V_tx(t) = V_inc + V_ref  (送信端での観測)
  ④ 受信端波形 V_rx(t) = (1+Γ) × V_inc(t − TD)  (負荷端での観測)

パラメータを TD = 1 ns / f = 500 MHz / Γ = +1（開放端）に設定し
「ちょうど 2TD = 2 ns = 1周期後」に反射が到達する様子を見やすくしている。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# ─────────────────────────────────────────────
# パラメータ
# ─────────────────────────────────────────────
Z1 = 50.0  # TL1 特性インピーダンス [Ω]
Z2 = 100.0  # TL2 特性インピーダンス [Ω]
ZL = 50.0  # 負荷インピーダンス [Ω]
RS = 50.0  # 信号源内部抵抗 [Ω]
VBIAS = 1.0  # 終端バイアス電圧 [V]
VS = 1.0  # 信号源電圧振幅 [V]
FREQ_HZ = 500e6  # 周波数 [Hz]  → 周期 T = 2 ns
TD1_NS = 1.0  # TL1 片道遅延 [ns]
TD2_NS = 1.0  # TL2 片道遅延 [ns]

# 送信端（Rs と Z1 の分圧）で入射波振幅が決まる
AMPLITUDE = VS * Z1 / (RS + Z1)  # = 0.5 V

# 接続点（Z1→Z2）での反射・透過係数
GAMMA_J = (Z2 - Z1) / (Z2 + Z1)  # 前方反射係数
T_J_FWD = 2 * Z2 / (Z2 + Z1)  # 前方透過係数
GAMMA_J_BWD = (Z1 - Z2) / (Z1 + Z2)  # 後方反射係数 = −GAMMA_J
T_J_BWD = 2 * Z1 / (Z2 + Z1)  # 後方透過係数

# 負荷（ZL, Z2 不整合）での反射係数
GAMMA_L = (ZL - Z2) / (ZL + Z2)

FIX_END_NS = 4.0  # 前半フラット終了時刻 [ns]
SIN_END_NS = 12.0  # Sin区間終了時刻 [ns]
BURST_END_NS = 16.0  # バースト終了 [ns]

T_PERIOD_NS = 1.0 / FREQ_HZ * 1e9  # = 2 ns
FIRST_REF_NS = 2 * TD1_NS  # 接続点反射が源端に戻る時刻 [ns]
OBS_ARRIVE_NS = TD1_NS + TD2_NS  # 信号が観測点に到達する時刻 [ns]
LOAD_REF_NS = 2 * (TD1_NS + TD2_NS)  # 負荷反射が源端に戻る時刻 [ns]

PLOT_END_NS = 22.0  # 表示終了 [ns]

N_POINTS = 8000

# ─────────────────────────────────────────────
# 波形関数
# ─────────────────────────────────────────────
t_ns = np.linspace(0, PLOT_END_NS, N_POINTS)
t = t_ns * 1e-9


def v_inc(t_arr: np.ndarray) -> np.ndarray:
    """入射波: 前半フラット → Sin(4周期) → 後半フラット"""
    fix_end = FIX_END_NS * 1e-9
    sin_end = SIN_END_NS * 1e-9
    burst_end = BURST_END_NS * 1e-9
    flat1_gate = ((t_arr >= 0) & (t_arr < fix_end)).astype(float)
    sin_gate = ((t_arr >= fix_end) & (t_arr < sin_end)).astype(float)
    flat2_gate = (t_arr >= sin_end).astype(float)  # バースト終了後もずっと AMPLITUDE
    flat1_part = AMPLITUDE * flat1_gate
    # 0V〜AMPLITUDE で振れる単極 Sin: A/2 + A/2*cos(ωt)
    # phase π/2 で開始 → cos(0)=1 なので t=fix_end で AMPLITUDE から始まり 0V まで下がる
    sin_part = (
        AMPLITUDE / 2 + AMPLITUDE / 2 * np.sin(2 * np.pi * FREQ_HZ * (t_arr - fix_end) + np.pi / 2)
    ) * sin_gate
    flat2_part = AMPLITUDE * flat2_gate
    return flat1_part + sin_part + flat2_part


def _delayed(t_arr: np.ndarray, delay_ns: float) -> np.ndarray:
    """V_inc を delay_ns [ns] だけ遅延させた波形"""
    d = delay_ns * 1e-9
    t_d = np.where(t_arr >= d, t_arr - d, 0.0)
    return v_inc(t_d) * (t_arr >= d)


def v_ref(t_arr: np.ndarray) -> np.ndarray:
    """送信端に戻る反射波: 接続点反射 + 負荷反射の重ね合わせ"""
    # 接続点反射: Γ_J × V_inc(t − 2·TD1)
    vr_j = GAMMA_J * _delayed(t_arr, 2 * TD1_NS)
    # 負荷反射が TL1 に透過: T_J_FWD·Γ_L·T_J_BWD × V_inc(t − 2·TD1 − 2·TD2)
    vr_L = T_J_FWD * GAMMA_L * T_J_BWD * _delayed(t_arr, 2 * TD1_NS + 2 * TD2_NS)
    return vr_j + vr_L


def v_rx(t_arr: np.ndarray) -> np.ndarray:
    """観測点(TL2終端, RL手前)の電圧: (1+Γ_L)·T_J_FWD × V_inc(t − TD1 − TD2)"""
    return (1 + GAMMA_L) * T_J_FWD * _delayed(t_arr, TD1_NS + TD2_NS)


vi = v_inc(t)
vr = v_ref(t)
vrx = v_rx(t)

# ─────────────────────────────────────────────
# 描画
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 1, 1, 1])
ax_ckt = fig.add_subplot(gs[0])
ax_inc = fig.add_subplot(gs[1])
ax_ref = fig.add_subplot(gs[2])
ax_rx = fig.add_subplot(gs[3])

# 波形段で x 軸共有
for ax in [ax_ref, ax_rx]:
    ax.sharex(ax_inc)

ZONE_NO_REF_COLOR = "#e8f5e9"
ZONE_WITH_REF_COLOR = "#fff3e0"


def shade_zones(ax):
    ax.axvspan(0, FIRST_REF_NS, color=ZONE_NO_REF_COLOR, alpha=0.7, zorder=0)
    ax.axvspan(FIRST_REF_NS, PLOT_END_NS, color=ZONE_WITH_REF_COLOR, alpha=0.5, zorder=0)
    ax.axvline(FIRST_REF_NS, color="dimgray", linestyle="--", linewidth=1.5, zorder=3)


# ── 回路図 ───────────────────────────────────────────
def draw_circuit(ax: plt.Axes) -> None:
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("① シミュレーション回路構成", fontsize=12, fontweight="bold")

    LINE_Y_TOP = 3.4
    LINE_Y_BOT = 2.0
    LINE_MID = (LINE_Y_TOP + LINE_Y_BOT) / 2

    # ── 信号源 (V_s ボックス)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.2, LINE_MID - 0.7),
            1.6,
            1.4,
            boxstyle="round,pad=0.15",
            facecolor="lightcyan",
            edgecolor="steelblue",
            linewidth=2,
        )
    )
    ax.text(1.0, LINE_MID + 0.2, "$V_s$", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(1.0, LINE_MID - 0.35, "入力波形", ha="center", va="center", fontsize=8.5)

    # ── Vs → Rs 接続線
    ax.plot([1.8, 2.4], [LINE_MID, LINE_MID], "-", color="black", linewidth=2.0)

    # ── Rs ボックス
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (2.4, LINE_MID - 0.45),
            1.4,
            0.9,
            boxstyle="round,pad=0.1",
            facecolor="lightyellow",
            edgecolor="darkgoldenrod",
            linewidth=2,
        )
    )
    ax.text(3.1, LINE_MID, f"$R_s={RS:.0f}\\,\\Omega$", ha="center", va="center", fontsize=9)

    # ── Rs → 伝送線路接続線
    ax.plot([3.8, 4.5], [LINE_MID, LINE_MID], "-", color="black", linewidth=2.0)

    # ── TL1（Z1=50Ω）
    TL1_X0, TL1_X1 = 4.5, 7.3
    for y in [LINE_Y_TOP - 0.4, LINE_Y_BOT + 0.4]:
        ax.plot([TL1_X0, TL1_X1], [y, y], "-", color="black", linewidth=2.5)
    ax.plot(
        [TL1_X0, TL1_X0], [LINE_Y_BOT + 0.4, LINE_Y_TOP - 0.4], "-", color="black", linewidth=1.5
    )
    ax.plot(
        [TL1_X1, TL1_X1], [LINE_Y_BOT + 0.4, LINE_Y_TOP - 0.4], "-", color="black", linewidth=1.8
    )
    ax.text(
        (TL1_X0 + TL1_X1) / 2,
        LINE_MID,
        f"TL1\n$Z_1={Z1:.0f}\\,\\Omega$\n$T_{{D1}}={TD1_NS}\\,ns$",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.9
        ),
    )

    # ── TL2（Z2=60Ω）
    TL2_X0, TL2_X1 = 7.3, 10.2
    for y in [LINE_Y_TOP - 0.4, LINE_Y_BOT + 0.4]:
        ax.plot([TL2_X0, TL2_X1], [y, y], "-", color="navy", linewidth=2.5)
    ax.plot(
        [TL2_X0, TL2_X0], [LINE_Y_BOT + 0.4, LINE_Y_TOP - 0.4], "-", color="black", linewidth=1.8
    )
    ax.plot(
        [TL2_X1, TL2_X1],
        [LINE_Y_BOT + 0.4, LINE_Y_TOP - 0.4],
        ":",
        color="darkorchid",
        linewidth=1.5,
    )
    ax.text(
        (TL2_X0 + TL2_X1) / 2,
        LINE_MID,
        f"TL2\n$Z_2={Z2:.0f}\\,\\Omega$\n$T_{{D2}}={TD2_NS}\\,ns$",
        ha="center",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd", edgecolor="steelblue", alpha=0.9),
    )

    # ── 接続点マーカー（TL1→TL2）
    ax.plot(TL1_X1, LINE_MID, "D", color="darkorange", markersize=7, zorder=6)
    ax.annotate(
        f"$\\Gamma_J={GAMMA_J:+.3f}$\n接続点",
        xy=(TL1_X1, LINE_MID),
        xytext=(TL1_X1 - 0.5, 4.9),
        fontsize=8,
        color="darkorange",
        arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff8e1", edgecolor="orange", alpha=0.9),
    )

    # ── 観測点（伝送線路出口 ＝ RL 手前）
    ax.plot(10.2, LINE_MID, "o", color="darkorchid", markersize=11, zorder=5)
    ax.annotate(
        "観測点 $V_{obs}$\n(受信端)",
        xy=(10.2, LINE_MID),
        xytext=(8.8, 4.6),
        fontsize=9.5,
        color="darkorchid",
        arrowprops=dict(arrowstyle="-|>", color="darkorchid", lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="#f3e5f5", edgecolor="darkorchid", alpha=0.95
        ),
    )

    # ── RL 接続線
    ax.plot([10.2, 10.8], [LINE_MID, LINE_MID], "-", color="black", linewidth=2.0)

    # ── RL ボックス
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (10.8, LINE_MID - 0.45),
            1.4,
            0.9,
            boxstyle="round,pad=0.1",
            facecolor="lightyellow",
            edgecolor="darkgoldenrod",
            linewidth=2,
        )
    )
    ax.text(11.5, LINE_MID, f"$R_L={ZL:.0f}\\,\\Omega$", ha="center", va="center", fontsize=9)

    # ── RL → GND（垂直線で接地）
    ax.plot([12.2, 12.6], [LINE_MID, LINE_MID], "-", color="black", linewidth=2.0)
    # 接地シンボル
    for i, w in enumerate([0.35, 0.22, 0.1]):
        y_gnd = LINE_MID - 0.25 - i * 0.18
        ax.plot([12.6 - w, 12.6 + w], [y_gnd, y_gnd], "-", color="black", linewidth=2.0 - i * 0.5)
    ax.plot([12.6, 12.6], [LINE_MID, LINE_MID - 0.25], "-", color="black", linewidth=2.0)

    # ── 入射波矢印（TL1 と TL2 内）
    ax.annotate(
        "",
        xy=(9.4, LINE_Y_TOP),
        xytext=(5.0, LINE_Y_TOP),
        arrowprops=dict(arrowstyle="-|>", color="royalblue", lw=2.0),
    )
    ax.text(
        7.2, LINE_Y_TOP + 0.25, "入射波 $V_{inc}(t)$", color="royalblue", ha="center", fontsize=9
    )

    # ── 反射波矢印（接続点反射 + 負荷反射）
    ax.annotate(
        "",
        xy=(5.0, LINE_Y_BOT),
        xytext=(9.4, LINE_Y_BOT),
        arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.0),
    )
    ax.text(
        7.2,
        LINE_Y_BOT - 0.3,
        f"反射波（接続点：$\\Gamma_J={GAMMA_J:+.3f}$、負荷：$\\Gamma_L={GAMMA_L:+.3f}$）",
        color="crimson",
        ha="center",
        fontsize=8.5,
    )

    # ── 式
    ax.text(
        7.5,
        0.55,
        f"$\\Gamma_J=\\frac{{Z_2-Z_1}}{{Z_2+Z_1}}={GAMMA_J:+.3f}$",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e1", edgecolor="orange", alpha=0.9),
    )
    ax.text(
        12.2,
        0.55,
        f"$\\Gamma_L=\\frac{{Z_L-Z_2}}{{Z_L+Z_2}}={GAMMA_L:+.3f}$",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="green", alpha=0.9),
    )


draw_circuit(ax_ckt)
ax = ax_inc
shade_zones(ax)
ax.plot(t_ns, vi, color="royalblue", linewidth=2.2, zorder=4, label="入射波 $V_{inc}(t)$")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("電圧 (V)", fontsize=11)
ax.set_title("② 入射波  $V_{inc}(t)$ ― 前半フラット / 後半 Sin", fontsize=12, fontweight="bold")
ax.set_ylim(-0.15, 0.85)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

# フラット→Sin 切替ライン
ax.axvline(FIX_END_NS, color="royalblue", linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(
    FIX_END_NS + 0.1, -0.1, f"Sin 開始\n$t={FIX_END_NS:.0f}$ ns", color="royalblue", fontsize=8.5
)

# Sin→後半フラット切替ライン
ax.axvline(SIN_END_NS, color="royalblue", linestyle=":", linewidth=1.5, alpha=0.8)
ax.text(
    SIN_END_NS + 0.1, -0.1, f"Sin 終了\n$t={SIN_END_NS:.0f}$ ns", color="royalblue", fontsize=8.5
)

# 区間ラベル
ax.text(
    TD1_NS,
    0.72,
    "反射未到達\n(クリーンな波形)",
    ha="center",
    va="center",
    fontsize=10,
    color="#1b5e20",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor=ZONE_NO_REF_COLOR, edgecolor="#388e3c", alpha=0.95
    ),
)
ax.text(
    (FIRST_REF_NS + PLOT_END_NS) / 2,
    0.72,
    "反射が届く区間\n(この後で波形が変化)",
    ha="center",
    va="center",
    fontsize=10,
    color="#e65100",
    bbox=dict(
        boxstyle="round,pad=0.3", facecolor=ZONE_WITH_REF_COLOR, edgecolor="#f57c00", alpha=0.95
    ),
)

# バースト終了マーカ
ax.axvline(BURST_END_NS, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(
    BURST_END_NS + 0.1, -0.1, f"バースト終了\n$t={BURST_END_NS:.0f}$ ns", color="gray", fontsize=8.5
)

# ── 反射波 ───────────────────────────────────────────
ax = ax_ref
shade_zones(ax)
ax.plot(
    t_ns,
    vr,
    color="crimson",
    linewidth=2.2,
    zorder=4,
    label=f"反射波 (TL1出口):  $\\Gamma_J={GAMMA_J:+.3f}$、$\\Gamma_L={GAMMA_L:+.3f}$",
)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("電圧 (V)", fontsize=11)
ax.set_title(
    f"③ 送信端に戻る反射波  接続点反射($t={FIRST_REF_NS:.0f}$ ns) + 負荷反射($t={LOAD_REF_NS:.0f}$ ns)",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylim(-0.15, 0.25)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

# 「ここで反射波が届く」矢印
ax.annotate(
    f"$t = 2T_{{D1}} = {FIRST_REF_NS:.0f}$ ns\n接続点反射到達",
    xy=(FIRST_REF_NS, 0.0),
    xytext=(FIRST_REF_NS + 1.5, 0.15),
    fontsize=9,
    color="crimson",
    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", edgecolor="crimson", alpha=0.9),
)
ax.text(
    TD1_NS * 0.7,
    0.08,
    "反射波なし\n(ゼロ)",
    ha="center",
    va="center",
    fontsize=9.5,
    color="gray",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="lightgray", alpha=0.9),
)

# 負荷反射到達マーカー
ax.axvline(LOAD_REF_NS, color="darkgreen", linestyle=":", linewidth=1.5, zorder=4)
ax.annotate(
    f"$t={LOAD_REF_NS:.0f}$ ns\n負荷反射到達",
    xy=(LOAD_REF_NS, 0.0),
    xytext=(LOAD_REF_NS + 1.0, 0.15),
    fontsize=9,
    color="darkgreen",
    arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="green", alpha=0.9),
)

# 2TD1 垂直ライン ラベル（全体で共有）
ax_ckt.text(
    FIRST_REF_NS + 0.07,
    -0.1,
    f"←  接続点反射到達\n   $t=2T_{{D1}}={FIRST_REF_NS:.0f}$ ns",
    color="dimgray",
    fontsize=8.5,
)

# ── 受信端波形 ────────────────────────────────────────
ax = ax_rx

# 受信端では反射なし区間と反射あり区間の境界は TD（入射波が届く時刻）
ax.axvspan(0, OBS_ARRIVE_NS, color="#fce4ec", alpha=0.7, zorder=0)  # 信号未到達
ax.axvspan(OBS_ARRIVE_NS, PLOT_END_NS, color="#e8eaf6", alpha=0.5, zorder=0)  # 信号到達済み
ax.axvline(OBS_ARRIVE_NS, color="dimgray", linestyle="--", linewidth=1.5, zorder=3)

# 入射波（薄く参考）
ax.plot(
    t_ns,
    vi,
    color="royalblue",
    linewidth=1.2,
    alpha=0.3,
    linestyle="--",
    zorder=3,
    label="入射波のみ（参考, 遅延なし表示）",
)

# 受信端波形
ax.plot(
    t_ns,
    vrx,
    color="darkorchid",
    linewidth=2.5,
    zorder=4,
    label=f"観測点 $V_{{obs}} = (1+\\Gamma_L)\\cdot T_{{J,fwd}}\\cdot V_{{inc}}(t-T_{{D1}}-T_{{D2}})$  "
    f"$振幅係数 = {(1+GAMMA_L)*T_J_FWD:.3f}$",
)

# 振幅増加を塗りつぶし
rx_mask = t_ns >= OBS_ARRIVE_NS
ax.fill_between(t_ns, 0, vrx, where=rx_mask, color="#ce93d8", alpha=0.25, zorder=2)

ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("電圧 (V)", fontsize=11)
ax.set_xlabel("時間 (ns)", fontsize=11)
ax.set_title(
    f"④ 観測点(TL2終端)波形  $V_{{obs}} = (1+\\Gamma_L)\\cdot T_{{J,fwd}}\\cdot V_{{inc}}(t-T_{{D1}}-T_{{D2}})$  "
    f"  振幅係数 $= {(1+GAMMA_L)*T_J_FWD:.3f}$",
    fontsize=11,
    fontweight="bold",
)
ax.set_ylim(-0.15, 1.3)
ax.legend(loc="upper right", fontsize=9.5, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))

# 信号未到達ラベル
ax.text(
    OBS_ARRIVE_NS * 0.5,
    0.25,
    "信号未到達\n(ゼロ)",
    ha="center",
    va="center",
    fontsize=9,
    color="#880e4f",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#fce4ec", edgecolor="#e91e63", alpha=0.9),
)

# 振幅2倍アノテーション
peak_idx = np.argmax(vrx[t_ns >= OBS_ARRIVE_NS + 0.1])
peak_t = t_ns[t_ns >= OBS_ARRIVE_NS + 0.1][peak_idx]
peak_v = vrx[t_ns >= OBS_ARRIVE_NS + 0.1][peak_idx]
obs_amplitude = (1 + GAMMA_L) * T_J_FWD * AMPLITUDE
ax.annotate(
    f"振幅 = {obs_amplitude:.3f} V\n入射波の {(1+GAMMA_L)*T_J_FWD:.3f} 倍",
    xy=(peak_t, peak_v),
    xytext=(peak_t + 1.5, peak_v - 0.1),
    fontsize=9.5,
    color="darkorchid",
    arrowprops=dict(arrowstyle="-|>", color="darkorchid", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3e5f5", edgecolor="darkorchid", alpha=0.95),
)

ax.text(
    OBS_ARRIVE_NS + 0.1,
    -0.12,
    f"$t=T_{{D1}}+T_{{D2}}={OBS_ARRIVE_NS}$ ns 以降:\n観測点に信号が届き始める",
    color="#4527a0",
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8eaf6", edgecolor="#5e35b1", alpha=0.9),
)

# ── 全体タイトル ────────────────────────────
fig.suptitle(
    f"バースト Sin 波形と反射の影響  (Ⅱ段伝送路: TL1→TL2)\n"
    f"$Z_1={Z1:.0f}\\,Ω$,  $Z_2={Z2:.0f}\\,Ω$,  $Z_L={ZL:.0f}\\,Ω$,  $R_s={RS:.0f}\\,Ω$,  "
    f"$\\Gamma_J={GAMMA_J:+.3f}$,  $\\Gamma_L={GAMMA_L:+.3f}$,  "
    f"$f={FREQ_HZ/1e6:.0f}\\,MHz$,  $T_{{D1}}={TD1_NS}\\,ns$,  $T_{{D2}}={TD2_NS}\\,ns$",
    fontsize=12,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", edgecolor="mediumpurple", alpha=0.85),
)

# ── 保存 ────────────────────────────────────
out_path = Path(__file__).with_name("reflection_burst_demo.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")
