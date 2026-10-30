"""
標本サイズ n に対する標本平均の確率分布シミュレーター
========================================================================================
実行方法:
    pip install streamlit numpy plotly
    streamlit run clt_theory_3modes_app.py
"""

import math
import time
from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. ページ設定 & カスタムCSS（横画面2カラム固定・見切れ防止・サイドバー対応）
# ============================================================================
st.set_page_config(
    page_title="中心極限定理シミュレーター（標本平均の確率分布）",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* 1. Streamlitの上部ヘッダーバーを透明化 */
    header[data-testid="stHeader"] {
        background: transparent !important;
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

    /* PC横画面レイアウト：メイン領域の左右2カラムを折り返さず横並びで固定 */
    .main [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
    }
    .main [data-testid="column"] {
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
# 2. 3つの分布モード定義 & セッション状態の初期化
# ============================================================================
MODE_BERNOULLI = "ベルヌーイ分布B(1,0.6)"
MODE_UNIFORM = "一様分布（さいころ）"
MODE_U_SHAPE = "U字型（1,2,3,4,5）"

DIST_MODES: Dict[str, Dict] = {
    MODE_BERNOULLI: {
        "title": "くじ引き",
        "pop_title": "ベルヌーイ分布",
        "values": np.array([0, 1], dtype=int),
        "probs": np.array([0.40, 0.60], dtype=float),
        "bar_colors": ["#94a3b8", "#10b981"],
        "border_colors": ["#64748b", "#059669"],
        "pop_y_max": 0.80,
        "pop_y_dtick": 0.20,
        "fixed_x_range": [-0.25, 1.25],
        "fixed_x_dtick": 0.2,
        "fixed_label": "横軸 0〜1 固定",
    },
    MODE_UNIFORM: {
        "title": "さいころ",
        "pop_title": "一様分布",
        "values": np.array([1, 2, 3, 4, 5, 6], dtype=int),
        "probs": np.array([1.0 / 6.0] * 6, dtype=float),
        "bar_colors": ["#60a5fa"] * 6,
        "border_colors": ["#2563eb"] * 6,
        "pop_y_max": 0.30,
        "pop_y_dtick": 0.05,
        "fixed_x_range": [0.5, 6.5],
        "fixed_x_dtick": 0.5,
        "fixed_label": "横軸 1〜6 固定",
    },
    MODE_U_SHAPE: {
        "title": "こんな分布でも...",
        "pop_title": "U字型分布",
        "values": np.array([1, 2, 3, 4, 5], dtype=int),
        "probs": np.array([0.31, 0.16, 0.06, 0.16, 0.31], dtype=float),
        "bar_colors": ["#6366f1", "#94a3b8", "#cbd5e1", "#94a3b8", "#10b981"],
        "border_colors": ["#4338ca", "#64748b", "#94a3b8", "#64748b", "#059669"],
        "pop_y_max": 0.40,
        "pop_y_dtick": 0.10,
        "fixed_x_range": [0.5, 5.5],
        "fixed_x_dtick": 0.5,
        "fixed_label": "横軸 1〜5 固定",
    },
}

# 連続実行の更新間隔（秒）
SLEEP_INTERVAL = 0.12


def init_session_state() -> None:
    """セッション状態を初期化する"""
    if "dist_mode" not in st.session_state:
        st.session_state.dist_mode = MODE_BERNOULLI
    if "n" not in st.session_state:
        st.session_state.n = 1
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "show_normal_curve" not in st.session_state:
        st.session_state.show_normal_curve = True
    if "x_axis_mode" not in st.session_state:
        st.session_state.x_axis_mode = "fixed"


init_session_state()


def get_population_stats(mode_key: str) -> Tuple[float, float, float]:
    """選択中の母集団の母平均 μ, 母分散 σ^2, 母標準偏差 σ を返す"""
    cfg = DIST_MODES[mode_key]
    vals = cfg["values"]
    probs = cfg["probs"]
    mean = float(np.sum(vals * probs))
    var = float(np.sum(((vals - mean) ** 2) * probs))
    std = math.sqrt(var)
    return mean, var, std


# ============================================================================
# 3. サイドバー（3モード切替）
# ============================================================================
with st.sidebar:
    selected_mode = st.radio(
        "母集団分布",
        options=[MODE_BERNOULLI, MODE_UNIFORM, MODE_U_SHAPE],
        index=[MODE_BERNOULLI, MODE_UNIFORM, MODE_U_SHAPE].index(st.session_state.dist_mode),
        label_visibility="collapsed",
    )

    if selected_mode != st.session_state.dist_mode:
        st.session_state.dist_mode = selected_mode
        st.session_state.is_running = False
        st.session_state.n = 1
        st.rerun()

    pop_mean, pop_var, pop_std = get_population_stats(st.session_state.dist_mode)


# ============================================================================
# 4. 標本平均の確率分布（厳密な畳み込み）計算ロジック
# ============================================================================
@st.cache_data
def compute_sample_mean_distribution(mode_key: str, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    繰り返し二乗法による畳み込み（Convolution）を用いて、
    標本サイズ n における標本平均 X_bar の厳密な確率分布を高速に計算する。
    """
    cfg = DIST_MODES[mode_key]
    vals = cfg["values"]
    probs = cfg["probs"]

    result = np.array([1.0], dtype=float)
    base = probs.copy()
    exp = n

    while exp > 0:
        if exp % 2 == 1:
            result = np.convolve(result, base)
        base = np.convolve(base, base)
        exp //= 2

    # 浮動小数点誤差を補正して確率の総和を1に正規化
    result = result / np.sum(result)

    # n個の和 S_n は n*v_min 〜 n*v_max の整数値をとるため、標本平均は S_n / n
    v_min, v_max = int(vals[0]), int(vals[-1])
    sums = np.arange(n * v_min, n * v_max + 1, dtype=float)
    x_bars = sums / n
    return x_bars, result


# ============================================================================
# 5. HTML & チャート描画ヘルパー
# ============================================================================
def render_n_banner_html(n: int) -> str:
    """左カラム上部の「標本サイズ n」バナーHTML（中央揃え）"""
    return f"""
    <div class="result-banner">
        <div class="result-label">標本の大きさ</div>
        <div class="result-val-emerald">{n:,}</div>
    </div>
    """


def create_population_figure(mode_key: str) -> go.Figure:
    """左カラム：選択中の母集団の確率分布グラフ"""
    cfg = DIST_MODES[mode_key]
    vals = cfg["values"]
    probs = cfg["probs"]

    text_vals = [f"{p:.3f}" if mode_key == MODE_UNIFORM else f"{p:.2f}" for p in probs]
    bar_w = 0.38 if len(vals) == 2 else 0.55

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=vals,
            y=probs,
            width=bar_w,
            marker=dict(
                color=cfg["bar_colors"],
                line=dict(color=cfg["border_colors"], width=1.4),
            ),
            text=text_vals,
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            hovertemplate="値 %{x}<br>確率: <b>%{y:.4f}</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=42, r=20, t=20, b=36),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(
            range=cfg["fixed_x_range"],
            tickmode="array",
            tickvals=list(vals),
            ticktext=[str(v) for v in vals],
            fixedrange=True,
            gridcolor="#f8fafc",
            linecolor="#cbd5e1",
            tickfont=dict(size=12, color="#334155"),
        ),
        yaxis=dict(
            range=[0, cfg["pop_y_max"]],
            dtick=cfg["pop_y_dtick"],
            tickformat=".2f",
            fixedrange=True,
            gridcolor="#f1f5f9",
            zeroline=True,
            zerolinecolor="#cbd5e1",
            tickfont=dict(size=11, color="#64748b"),
        ),
    )
    return fig


def create_sample_mean_dist_figure(mode_key: str, n: int) -> go.Figure:
    """右カラム：標本サイズ n に対する標本平均 X_bar の確率分布グラフ"""
    cfg = DIST_MODES[mode_key]
    vals = cfg["values"]
    v_min, v_max = float(vals[0]), float(vals[-1])
    span = v_max - v_min

    x_bars, probs = compute_sample_mean_distribution(mode_key, n)
    mean, _, pop_std = get_population_stats(mode_key)

    std_xbar = pop_std / math.sqrt(n)
    step_w = 1.0 / n  # 隣り合う標本平均の間隔 (1/n)

    # 正規分布近似によるピーク確率 (刻み幅 1/n × 最大確率密度)
    theo_peak = step_w / (std_xbar * math.sqrt(2.0 * math.pi))
    max_prob = float(np.max(probs))

    # n が変わっても山が潰れないよう、縦軸の上限を現在のピークに合わせてスケーリング
    y_max = max(max_prob, theo_peak) * 1.18

    # 横軸の表示範囲（全体固定 または μ±4σ拡大）
    if st.session_state.x_axis_mode == "zoom" and n >= 4:
        half_span = max(4.0 * std_xbar, 0.02 * span)
        x_min = max(v_min - 0.15 * span, mean - half_span)
        x_max = min(v_max + 0.15 * span, mean + half_span)
        x_dtick = None
    else:
        x_min, x_max = cfg["fixed_x_range"]
        x_dtick = cfg["fixed_x_dtick"] if n > 1 else 1.0

    # 表示範囲内のデータのみ抽出
    mask = (x_bars >= x_min - step_w) & (x_bars <= x_max + step_w)
    vis_x = x_bars[mask]
    vis_p = probs[mask]

    if n == 1:
        bar_width = 0.38 if len(vals) == 2 else 0.55
    else:
        bar_width = step_w * 0.86

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
# 6. ヘッダー & トップコントロールバー
# ============================================================================
current_cfg = DIST_MODES[st.session_state.dist_mode]

st.markdown(
    f"""
    <div class="sim-header">
        <div class="sim-title">
            {current_cfg["title"]}
        </div>
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
    if st.button("+5", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n += 5

with ctrl_cols[3]:
    if st.button("+10", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n += 10

with ctrl_cols[4]:
    if st.button("リセット", use_container_width=True):
        st.session_state.is_running = False
        st.session_state.n = 1
        st.rerun()


# ============================================================================
# 7. メイン2カラムレイアウト（左: 母集団の確率分布 / 右: 標本平均の確率分布）
# ============================================================================
col_left, col_right = st.columns([4.2, 6.8], gap="large")

# ---------------- 左カラム: 標本サイズ n & 母集団の確率分布 ----------------
with col_left:
    with st.container(border=True):
        st.markdown(
            render_n_banner_html(st.session_state.n),
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="section-title">{current_cfg["pop_title"]}</div>',
            unsafe_allow_html=True,
        )
        pop_fig = create_population_figure(st.session_state.dist_mode)
        st.plotly_chart(pop_fig, use_container_width=True, config={"displayModeBar": False})


# ---------------- 右カラム: 標本平均の確率分布 ----------------
with col_right:

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

        mean_dist_fig = create_sample_mean_dist_figure(
            st.session_state.dist_mode,
            st.session_state.n,
        )
        st.plotly_chart(mean_dist_fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# 8. 連続実行モードの自動更新ループ
# ============================================================================
if st.session_state.is_running:
    time.sleep(SLEEP_INTERVAL)
    st.session_state.n += 1
    st.rerun()
