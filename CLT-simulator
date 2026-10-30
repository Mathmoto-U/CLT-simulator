"""
標本サイズ n に対する標本平均の確率分布（U字型母集団）シミュレーター (Streamlit版)
========================================================================================
実行方法:
    pip install streamlit numpy plotly
    streamlit run u_dist_theory_app.py
"""

import math
import time
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. ページ設定 & カスタムCSS（横画面2カラム固定・見切れ防止）
# ============================================================================
st.set_page_config(
    page_title="中心極限定理シミュレーター（標本平均の確率分布）",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* 1. Streamlitの上部ヘッダーバーを透明化＆高さを最小化 */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }

    /* 2. ページ全体のスクロールを許可 */
    html, body, [data-testid="stAppViewContainer"], .main {
        overflow-y: auto !important;
    }

    /* 3. コンテナ上部の余白を確保して見切れを防止 */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* サイドバー開閉ボタンを非表示 */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* PC横画面レイアウト：左右2カラムを折り返さず横並びで固定 */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
    }
    [data-testid="column"] {
        min-width: 0 !important;
    }

    /* ヘッダーバナー */
    .sim-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 12px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .sim-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.3;
    }
    .sim-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin: 3px 0 0 0;
    }

    /* 左カラム：標本サイズ n バナー（中央揃え） */
    .result-banner {
        background: linear-gradient(90deg, #ecfdf5 0%, #f0fdfa 50%, #ecfdf5 100%);
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 14px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .result-label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 2px;
    }
    .result-val-emerald {
        font-size: 2.1rem;
        font-weight: 800;
        color: #059669;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        line-height: 1.1;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #334155;
        text-align: center;
        margin: 4px 0 6px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# 2. 定数・母集団の確率分布（1〜5のU字型分布）・セッション状態の初期化
# ============================================================================
VALUES = np.array([1, 2, 3, 4, 5], dtype=int)

# 1 - (反復試行の確率) 型のU字分布: 両外側(1, 5)が31%、内側(2, 4)が16%、中央(3)が6%
PROBS = np.array([0.31, 0.16, 0.06, 0.16, 0.31], dtype=float)
# ※純粋な (1 - C(4,k)/16) / 4 を使う場合は以下の行のコメントを外してください:
# PROBS = np.array([15/64, 12/64, 10/64, 12/64, 15/64], dtype=float)

# 母平均 μ と 母分散 σ^2
POP_MEAN = float(np.sum(VALUES * PROBS))                          # 3.00
POP_VAR = float(np.sum(((VALUES - POP_MEAN) ** 2) * PROBS))       # 2.80
POP_STD = math.sqrt(POP_VAR)                                      # ≒ 1.6733

# 連続実行の更新間隔（秒）
SLEEP_INTERVAL = 0.12


def init_session_state() -> None:
    """セッション状態を初期化する"""
    if "n" not in st.session_state:
        st.session_state.n = 1  # 初期標本サイズ n = 1
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "show_normal_curve" not in st.session_state:
        st.session_state.show_normal_curve = True
    if "x_axis_mode" not in st.session_state:
        st.session_state.x_axis_mode = "fixed"  # "fixed" (1〜5固定) or "zoom" (±4σ拡大)


init_session_state()


# ============================================================================
# 3. 標本平均の確率分布（厳密な畳み込み）計算ロジック
# ============================================================================
@st.cache_data
def compute_sample_mean_distribution(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    繰り返し二乗法による畳み込み（Convolution）を用いて、
    標本サイズ n における標本平均 X_bar の厳密な確率分布を高速に計算する。
    """
    result = np.array([1.0], dtype=float)
    base = PROBS.copy()
    exp = n

    while exp > 0:
        if exp % 2 == 1:
            result = np.convolve(result, base)
        base = np.convolve(base, base)
        exp //= 2

    # 浮動小数点誤差を補正して確率の総和を1に正規化
    result = result / np.sum(result)

    # n個の和 S_n は n 〜 5n の整数値をとるため、標本平均は (n 〜 5n) / n
    sums = np.arange(n, 5 * n + 1, dtype=float)
    x_bars = sums / n
    return x_bars, result


# ============================================================================
# 4. HTML & チャート描画ヘルパー
# ============================================================================
def render_n_banner_html(n: int) -> str:
    """左カラム上部の「標本サイズ n」バナーHTML（中央揃え）"""
    return f"""
    <div class="result-banner">
        <div class="result-label">標本サイズ n</div>
        <div class="result-val-emerald">{n:,}</div>
    </div>
    """


def create_population_figure() -> go.Figure:
    """
    左カラム：母集団（1, 2, 3, 4, 5のU字型分布）の確率分布グラフ
    """
    bar_colors = ["#6366f1", "#94a3b8", "#cbd5e1", "#94a3b8", "#10b981"]
    border_colors = ["#4338ca", "#64748b", "#94a3b8", "#64748b", "#059669"]
    text_vals = [f"{p:.2f}" for p in PROBS]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=VALUES,
            y=PROBS,
            width=0.55,
            marker=dict(color=bar_colors, line=dict(color=border_colors, width=1.4)),
            text=text_vals,
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            hovertemplate="値 %{x}<br>確率: <b>%{y:.2f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=42, r=20, t=20, b=36),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
            fixedrange=True,
            gridcolor="#f8fafc",
            linecolor="#cbd5e1",
            tickfont=dict(size=12, color="#334155"),
        ),
        yaxis=dict(
            range=[0, 0.40],
            dtick=0.10,
            tickformat=".2f",
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
        ),
    )
    return fig


def create_sample_mean_dist_figure(n: int) -> go.Figure:
    """
    右カラム：標本サイズ n に対する標本平均 X_bar の確率分布グラフ
    """
    x_bars, probs = compute_sample_mean_distribution(n)

    mean = POP_MEAN
    std_xbar = POP_STD / math.sqrt(n)
    step_w = 1.0 / n  # 隣り合う標本平均の間隔 (1/n)

    # 正規分布近似によるピーク確率 (刻み幅 1/n × 最大確率密度)
    theo_peak = step_w / (std_xbar * math.sqrt(2.0 * math.pi))
    max_prob = float(np.max(probs))

    # n が変わっても山が潰れないよう、縦軸の上限を現在のピークに合わせてスケーリング
    y_max = max(max_prob, theo_peak) * 1.18

    # 横軸の表示範囲（1〜5固定 または μ±4σ拡大）
    if st.session_state.x_axis_mode == "zoom" and n >= 4:
        half_span = max(4.0 * std_xbar, 0.08)
        x_min = max(0.8, mean - half_span)
        x_max = min(5.2, mean + half_span)
        x_dtick = None
    else:
        x_min, x_max = 0.5, 5.5
        x_dtick = 0.5 if n > 1 else 1.0

    # 表示範囲内のデータのみ抽出（描画パフォーマンス最適化）
    mask = (x_bars >= x_min - step_w) & (x_bars <= x_max + step_w)
    vis_x = x_bars[mask]
    vis_p = probs[mask]

    bar_width = 0.55 if n == 1 else step_w * 0.86

    fig = go.Figure()

    # 1. 標本平均の確率分布（棒グラフ）
    fig.add_trace(
        go.Bar(
            x=vis_x,
            y=vis_p,
            width=bar_width,
            marker=dict(
                color="rgba(16, 185, 129, 0.82)",
                line=dict(
                    color="#059669",
                    width=1.2 if len(vis_x) <= 60 else (0.5 if len(vis_x) <= 200 else 0),
                ),
            ),
            hovertemplate=(
                "<b>標本平均: %{x:.3f}</b><br>"
                "確率: <b>%{y:.4f}</b>"
                "<extra></extra>"
            ),
        )
    )

    # 2. 正規分布 N(μ, σ^2/n) の理論曲線オーバーレイ
    if st.session_state.show_normal_curve:
        x_curve = np.linspace(x_min, x_max, 500)
        z_curve = (x_curve - mean) / std_xbar
        pdf_curve = (1.0 / (std_xbar * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z_curve**2)
        y_curve = step_w * pdf_curve

        fig.add_trace(
            go.Scatter(
                x=x_curve,
                y=y_curve,
                mode="lines",
                line=dict(color="#f59e0b", width=3.2),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=440,
        margin=dict(l=52, r=20, t=16, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(
            range=[x_min, x_max],
            dtick=x_dtick,
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            linecolor="#cbd5e1",
            tickfont=dict(size=11, color="#475569"),
        ),
        yaxis=dict(
            range=[0, y_max],
            tickformat=".3f" if y_max < 0.08 else ".2f",
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
        ),
        bargap=0.06,
        hovermode="closest",
    )

    return fig


# ============================================================================
# 5. ヘッダー & トップコントロールバー
# ============================================================================
st.markdown(
    """
    <div class="sim-header">
        <div class="sim-title">
            標本平均の確率分布シミュレーター（U字型母集団）
        </div>
        <p class="sim-subtitle">
            母集団がU字型分布であっても、標本サイズ n を大きくすると標本平均の確率分布は正規分布へ収束します（中心極限定理）。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

ctrl_cols = st.columns([1.2, 1.6, 1.0, 1.0, 1.0])

with ctrl_cols[0]:
    if st.button("nを+1", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n += 1

with ctrl_cols[1]:
    run_btn_label = "一時停止" if st.session_state.is_running else "連続実行"
    run_btn_type = "secondary" if st.session_state.is_running else "primary"
    if st.button(run_btn_label, type=run_btn_type, use_container_width=True):
        st.session_state.is_running = not st.session_state.is_running
        if st.session_state.is_running:
            st.session_state.n += 1

with ctrl_cols[2]:
    if st.button("+10", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n += 10

with ctrl_cols[3]:
    if st.button("+100", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n += 100

with ctrl_cols[4]:
    if st.button("リセット", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n = 1
        st.rerun()


# ============================================================================
# 6. メイン2カラムレイアウト（左: 母集団の確率分布 / 右: 標本平均の確率分布）
# ============================================================================
col_left, col_right = st.columns([4.2, 6.8], gap="large")

# ---------------- 左カラム: 標本サイズ n & 母集団の確率分布 ----------------
with col_left:
    with st.container(border=True):
        # 1. 標本サイズ n バナー（中央揃え）
        st.markdown(
            render_n_banner_html(st.session_state.n),
            unsafe_allow_html=True,
        )

        # 2. 母集団の確率分布（1〜5のU字型分布）
        st.markdown(
            '<div class="section-title">母集団の確率分布（U字型分布）</div>',
            unsafe_allow_html=True,
        )
        pop_fig = create_population_figure()
        st.plotly_chart(pop_fig, use_container_width=True, config={"displayModeBar": False})


# ---------------- 右カラム: 標本平均の確率分布 ----------------
with col_right:
    with st.container(border=True):
        X_MODE_LABELS = {
            "fixed": "横軸 1〜5 固定",
            "zoom": "分布周辺を拡大",
        }

        top_opt_cols = st.columns([6, 4])
        with top_opt_cols[0]:
            selected_x_mode = st.radio(
                "表示範囲の選択",
                options=["fixed", "zoom"],
                format_func=lambda x: X_MODE_LABELS[x],
                index=["fixed", "zoom"].index(st.session_state.x_axis_mode),
                horizontal=True,
            )
            if selected_x_mode != st.session_state.x_axis_mode:
                st.session_state.x_axis_mode = selected_x_mode
                st.rerun()

        with top_opt_cols[1]:
            st.write("")
            st.session_state.show_normal_curve = st.toggle(
                "正規分布の理論曲線を表示",
                value=st.session_state.show_normal_curve,
            )

        mean_dist_fig = create_sample_mean_dist_figure(st.session_state.n)
        st.plotly_chart(mean_dist_fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# 7. 連続実行モードの自動更新ループ
# ============================================================================
if st.session_state.is_running:
    time.sleep(SLEEP_INTERVAL)
    st.session_state.n += 1
    st.rerun()
