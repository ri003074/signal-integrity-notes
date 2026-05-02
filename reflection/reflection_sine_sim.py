"""
Sin波形を使った伝送線路の反射シミュレーション

送信端・受信端でのSin波形の観測を可視化する。
- 入射波 (V_inc)
- 反射波 (V_ref = Γ × V_inc(t - 2TD))
- 送信端での合成波形
- 受信端 (負荷端) での波形

物理モデル
----------
  V_inc(t)   = A * sin(2π f t)   ... 送信端から送り出す正弦波
  V_ref(t)   = Γ * V_inc(t - 2TD)  ... 線路を往復して戻ってくる反射波
  V_tx(t)    = V_inc(t) + V_ref(t) ... 送信端で観測される合成波形
  V_rx(t)    = (1+Γ) * V_inc(t-TD) ... 受信端(負荷端)での波形
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401

    _JAPANESE = Truee
except ImportError:
    _JAPANESE = False

# ──────────────────────────────────────────────────────────────
# パラメータ
# ───────────────────────────────────────────────────────ｎ───────
Z0 = 50.0  # 特性インピーダンス [Ω]
FREQ_HZ = 1e9  # 信号周波数 [Hz]
TD_NS = 0.5  # 片道伝播遅延 [ns]  (=0.5 ns → λ/4 at 500 MHz)
AMPLITUDE = 1.0  # 入射波振幅 [V]
BURST_CYCLES = 2.2  # バースト長 [周期]
FOCUS_END_NS = 3.2  # 見やすさのための表示上限 [ns]

# ZL と色
CASES = [
    (1e9, "開放端  ZL→∞,  Γ=+1", "red", "-"),
    (150.0, "ZL=150Ω,  Γ=+0.5", "darkorange", "--"),
    (50.0, "整合終端  ZL=Z0,  Γ=0", "green", "-"),
    (16.7, "ZL=16.7Ω,  Γ=−0.5", "blueviolet", "--"),
    (0.0, "短絡端  ZL=0,  Γ=−1", "purple", "-"),
]


def gamma(zl: float, z0: float = Z0) -> float:
    if zl == 0.0:
        return -1.0
    if zl >= 1e8:
        return +1.0
    return (zl - z0) / (zl + z0)


def make_time(n_periods: float = 4.0, n_points: int = 4000) -> np.ndarray:
    t_end = n_periods / FREQ_HZ
    return np.linspace(0.0, t_end, n_points)


def v_inc(t: np.ndarray) -> np.ndarray:
    period = 1.0 / FREQ_HZ
    burst_end = BURST_CYCLES * period
    burst_gate = ((t >= 0.0) & (t <= burst_end)).astype(float)
    return AMPLITUDE * np.sin(2 * np.pi * FREQ_HZ * t) * burst_gate


def v_tx(t: np.ndarray, gam: float) -> np.ndarray:
    """送信端(Source)での観測波形 = 入射 + 反射"""
    td = TD_NS * 1e-9
    delay = 2 * td
    return v_inc(t) + gam * v_inc(np.where(t >= delay, t - delay, 0.0)) * (t >= delay)


def v_rx(t: np.ndarray, gam: float) -> np.ndarray:
    """受信端(Load)での観測波形 = (1+Γ) × V_inc(t - TD)"""
    td = TD_NS * 1e-9
    return (1 + gam) * v_inc(np.where(t >= td, t - td, 0.0)) * (t >= td)


# ──────────────────────────────────────────────────────────────
# ① ブロック図
# ──────────────────────────────────────────────────────────────
def draw_block_diagram(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("① 送受信端の構成", fontsize=11, fontweight="bold")

    # 送信端
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.2, 1.5),
            1.6,
            1.4,
            boxstyle="round,pad=0.15",
            facecolor="lightblue",
            edgecolor="steelblue",
            linewidth=2,
        )
    )
    ax.text(1.0, 2.2, "信号源\nZs = Z0", ha="center", va="center", fontsize=9, fontweight="bold")

    # 伝送線路
    for y in [1.7, 2.8]:
        ax.plot([1.8, 7.5], [y, y], "-", color="black", linewidth=2.5)
    ax.text(
        4.65,
        2.25,
        f"伝送線路  Z0={Z0:.0f}Ω   TD={TD_NS} ns",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.85
        ),
    )

    # 受信端（負荷）
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (7.5, 1.5),
            1.6,
            1.4,
            boxstyle="round,pad=0.15",
            facecolor="mistyrose",
            edgecolor="red",
            linewidth=2,
        )
    )
    ax.text(8.3, 2.2, "負荷\nZL (可変)", ha="center", va="center", fontsize=9, fontweight="bold")

    # 観測点アノテーション
    ax.text(
        1.0,
        3.35,
        "①送信端\nV_tx = V_inc + V_ref",
        ha="center",
        va="bottom",
        fontsize=8,
        color="steelblue",
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="aliceblue", edgecolor="steelblue", alpha=0.9
        ),
    )
    ax.text(
        8.3,
        3.35,
        "②受信端\nV_rx = (1+Γ)·V_inc(t−TD)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="crimson",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="mistyrose", edgecolor="crimson", alpha=0.9),
    )

    # 入射波・反射波矢印
    ax.annotate(
        "",
        xy=(7.4, 3.0),
        xytext=(1.8, 3.0),
        arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.0),
    )
    ax.text(4.65, 3.25, "入射波 V_inc", color="blue", ha="center", fontsize=8)

    ax.annotate(
        "",
        xy=(1.8, 1.45),
        xytext=(7.4, 1.45),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0),
    )
    ax.text(4.65, 1.1, "反射波 V_ref = Γ · V_inc(t−2TD)", color="red", ha="center", fontsize=8)

    # Γ 式
    ax.text(
        5.0,
        4.3,
        "反射係数  Γ = (ZL − Z0) / (ZL + Z0)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", edgecolor="purple", alpha=0.9),
    )


# ──────────────────────────────────────────────────────────────
# ② 入射波・反射波の分離表示（単一ケース）
# ──────────────────────────────────────────────────────────────
def plot_incident_reflected(ax: plt.Axes, t_ns: np.ndarray, t: np.ndarray) -> None:
    ax.set_title(
        "② バースト波: 前半は反射なし / 後半は反射あり（開放端 Γ = +1）",
        fontsize=11,
        fontweight="bold",
    )
    gam = 1.0
    td = TD_NS * 1e-9
    vi = v_inc(t)
    # 反射波単体 = Γ × V_inc(t - 2TD)
    vr_only = gam * v_inc(np.where(t >= 2 * td, t - 2 * td, 0.0)) * (t >= 2 * td)
    v_total = vi + vr_only

    # 反射到達前後を色分け
    ax.axvspan(float(t_ns[0]), 2 * TD_NS, color="#e8f5e9", alpha=0.45, lw=0)
    ax.axvspan(2 * TD_NS, float(t_ns[-1]), color="#ffebee", alpha=0.35, lw=0)

    ax.plot(t_ns, vi, "-", color="royalblue", linewidth=2.0, label="入射波 V_inc(t)")
    ax.plot(
        t_ns,
        vr_only,
        "--",
        color="darkorange",
        linewidth=2.0,
        label=f"反射波 Γ·V_inc(t−2TD)  [Γ={gam}]",
    )
    ax.plot(t_ns, v_total, "-", color="black", linewidth=2.2, label="合成波 V_inc + V_ref")
    post_mask = t_ns >= (2 * TD_NS)
    ax.fill_between(
        t_ns,
        vi,
        v_total,
        where=post_mask,
        color="#f06292",
        alpha=0.25,
        label="反射による増分（2TD以降）",
    )
    ax.axvline(
        2 * TD_NS,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"反射波到達  t = 2TD = {2*TD_NS} ns",
    )
    ax.text(0.12, 1.28, "前半: 反射の影響なし", fontsize=9, color="#1b5e20")
    ax.text(2 * TD_NS + 0.08, 1.28, "後半: 反射が重なる", fontsize=9, color="#b71c1c")

    ax.set_xlabel("時間 (ns)")
    ax.set_ylabel("電圧 (V)")
    ax.set_xlim(0.0, min(FOCUS_END_NS, float(t_ns[-1])))
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")


# ──────────────────────────────────────────────────────────────
# ③ 送信端での合成波形（複数ケース）
# ──────────────────────────────────────────────────────────────
def plot_source_waveforms(ax: plt.Axes, t_ns: np.ndarray, t: np.ndarray) -> None:
    ax.set_title(
        "③ 送信端での観測波形  V_tx = V_inc + Γ·V_inc(t−2TD)", fontsize=11, fontweight="bold"
    )
    ax.plot(t_ns, v_inc(t), "k--", linewidth=1.2, alpha=0.4, label="入射波のみ (参考)")
    for zl, label, color, ls in CASES:
        gam = gamma(zl)
        ax.plot(t_ns, v_tx(t, gam), ls, color=color, linewidth=2.0, label=f"{label}")

    ax.axvline(
        2 * TD_NS,
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=f"反射波到達 t=2TD={2*TD_NS} ns",
    )
    ax.axvspan(float(t_ns[0]), 2 * TD_NS, color="#e8f5e9", alpha=0.25, lw=0)
    ax.axvspan(2 * TD_NS, float(t_ns[-1]), color="#ffebee", alpha=0.18, lw=0)
    ax.text(0.12, 1.9, "前半: 反射なし", fontsize=9, color="#1b5e20")
    ax.text(2 * TD_NS + 0.08, 1.9, "後半: 反射あり", fontsize=9, color="#b71c1c")
    ax.set_xlabel("時間 (ns)")
    ax.set_ylabel("電圧 (V)")
    ax.set_xlim(0.0, min(FOCUS_END_NS, float(t_ns[-1])))
    ax.set_ylim(-2.4, 2.4)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)


# ──────────────────────────────────────────────────────────────
# ④ 受信端(負荷)での波形（複数ケース）
# ──────────────────────────────────────────────────────────────
def plot_load_waveforms(ax: plt.Axes, t_ns: np.ndarray, t: np.ndarray) -> None:
    ax.set_title("④ 受信端(負荷)での波形  V_rx = (1+Γ)·V_inc(t−TD)", fontsize=11, fontweight="bold")
    ax.plot(t_ns, v_inc(t), "k--", linewidth=1.2, alpha=0.4, label="入射波のみ (参考)")
    for zl, label, color, ls in CASES:
        gam = gamma(zl)
        ax.plot(t_ns, v_rx(t, gam), ls, color=color, linewidth=2.0, label=f"{label}")

    ax.axvline(TD_NS, color="gray", linestyle=":", linewidth=1.2, label=f"信号到達 t=TD={TD_NS} ns")
    ax.set_xlabel("時間 (ns)")
    ax.set_ylabel("電圧 (V)")
    ax.set_xlim(0.0, min(FOCUS_END_NS, float(t_ns[-1])))
    ax.set_ylim(-2.4, 2.4)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)


# ──────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────
def main() -> None:
    plt.rcParams["axes.unicode_minus"] = False

    t = make_time(n_periods=5.0, n_points=5000)
    t_ns = t * 1e9  # 表示用 [ns]
    fig, axes = plt.subplots(4, 1, figsize=(16, 9), constrained_layout=True)

    draw_block_diagram(axes[0])
    plot_incident_reflected(axes[1], t_ns, t)
    plot_source_waveforms(axes[2], t_ns, t)
    plot_load_waveforms(axes[3], t_ns, t)

    fig.suptitle(
        f"Burst Sin波形の反射シミュレーション\n"
        f"Z0={Z0:.0f}Ω, f={FREQ_HZ/1e9:.1f} GHz, TD={TD_NS} ns",
        fontsize=13,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", edgecolor="purple", alpha=0.85),
    )

    out_path = Path(__file__).with_name("reflection_sine_sim.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
