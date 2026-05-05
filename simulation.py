"""
경사하강법 시뮬레이션 (Gradient Descent Simulation)
데이터: CSV (x, y, z) — x·y가 가중치 좌표, z가 비용값
파일 미선택 시 기본 합성 비볼록 함수(Himmelblau 변형) 사용
"""

import os
import platform
import numpy as np
import matplotlib
# 백엔드는 OS별로 자동 선택 (Mac=macosx, Windows/Linux=TkAgg 등)
if platform.system() == "Darwin":
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (projection='3d' 등록용)
import tkinter as tk
from tkinter import filedialog

# ── 폰트/축 스타일 ────────────────────────────────────────────────────────────
# 한글 폰트: OS별로 시스템 기본 한글 폰트 지정 (없으면 sans-serif fallback)
_KOR_FONT = {
    "Darwin":  "AppleGothic",
    "Windows": "Malgun Gothic",
    "Linux":   "NanumGothic",
}.get(platform.system(), "DejaVu Sans")
plt.rcParams['font.family']        = _KOR_FONT
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top']    = False
plt.rcParams['axes.spines.right']  = False
plt.rcParams['axes.edgecolor']     = '#d4d4d8'
plt.rcParams['axes.linewidth']     = 0.8
plt.rcParams['xtick.color']        = '#71717a'
plt.rcParams['ytick.color']        = '#71717a'
plt.rcParams['xtick.labelsize']    = 9
plt.rcParams['ytick.labelsize']    = 9

# ── 파라미터 ──────────────────────────────────────────────────────────────────
LR               = 0.005
MOMENTUM         = 0.97
N_EPOCHS         = 600
GRAD_CLIP_NORM   = 20.0
MC_GRID_SIZE     = 4        # 4x4 = 16 MC 시작점
MC_MARGIN_FRAC   = 0.125
CLIP_OVERSHOOT   = 0.0625
GLOBAL_REL_TOL   = 0.005
SPATIAL_REL_TOL  = 0.15
GRID_RES         = 160
ANIM_INTERVAL_MS = 320

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
C_BG          = "#fafaf9"   # figure 배경 (stone-50)
C_PANEL       = "#ffffff"   # axes 배경
C_TEXT        = "#18181b"   # 주 텍스트 (zinc-900)
C_MUTED       = "#71717a"   # 보조 텍스트 (zinc-500)
C_PRIMARY     = "#4f46e5"   # 시작/다음 (indigo-600) — 텍스트 색
C_HOVER       = "#f4f4f5"   # 버튼 hover 시 미세한 배경 (zinc-100)
C_SUCCESS     = "#10b981"   # 전역 도달 (emerald-500)
C_DANGER      = "#ef4444"   # 지역 수렴 (red-500)
C_AMBER       = "#f59e0b"   # Phase 1 공/궤적
C_STAR        = "#fbbf24"   # 전역 최솟값 별

# 차분한 2-hue 그라데이션 (slate → amber). RdYlGn 신호등 느낌 X
COST_CMAP = LinearSegmentedColormap.from_list("cost_calm", [
    "#1e293b",  # slate-800  (low cost — 어두운 골짜기)
    "#475569",  # slate-600
    "#94a3b8",  # slate-400
    "#e2e8f0",  # slate-200
    "#fef3c7",  # amber-100
    "#fbbf24",  # amber-400
    "#d97706",  # amber-600  (high cost — 밝은 봉우리)
])

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def pick_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="비용 함수 데이터 선택 (x, y, z 컬럼 CSV)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def load_xyz(filepath):
    with open(filepath, 'r') as f:
        header = [h.strip() for h in f.readline().split(',')]
    xi, yi, zi = header.index('x'), header.index('y'), header.index('z')
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return data[:, xi], data[:, yi], data[:, zi]

def fit_poly_surface(x, y, z, deg=4):
    def features(x_arr, y_arr):
        cols = []
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                cols.append(x_arr**i * y_arr**j)
        return np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(features(x, y), z, rcond=None)
    def cost_fn(w1, w2):
        w1_arr = np.asarray(w1, dtype=float)
        w2_arr = np.asarray(w2, dtype=float)
        shape = np.broadcast_shapes(w1_arr.shape, w2_arr.shape)
        w1_flat = np.broadcast_to(w1_arr, shape).ravel()
        w2_flat = np.broadcast_to(w2_arr, shape).ravel()
        result = features(w1_flat, w2_flat) @ coeffs
        if shape == ():
            return float(result[0])
        return result.reshape(shape)
    return cost_fn

# ── 기본 합성 함수 (Himmelblau 변형) ─────────────────────────────────────────
def default_cost_fn(w1, w2):
    return ((w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2) / 50

# ── 파일 선택 ─────────────────────────────────────────────────────────────────
filepath = pick_file()
if filepath and os.path.exists(filepath):
    try:
        xd, yd, zd = load_xyz(filepath)
        cost_fn   = fit_poly_surface(xd, yd, zd)
        W1_RANGE  = (xd.min(), xd.max())
        W2_RANGE  = (yd.min(), yd.max())
        DATA_LABEL = os.path.basename(filepath)
    except Exception as e:
        print(f"파일 로드 실패: {e} → 기본 함수 사용")
        cost_fn    = default_cost_fn
        W1_RANGE   = W2_RANGE = (-4.0, 4.0)
        DATA_LABEL = "기본 합성 함수 (Himmelblau 변형)"
else:
    cost_fn    = default_cost_fn
    W1_RANGE   = W2_RANGE = (-4.0, 4.0)
    DATA_LABEL = "기본 합성 함수 (Himmelblau 변형)"

W1_SPAN = W1_RANGE[1] - W1_RANGE[0]
W2_SPAN = W2_RANGE[1] - W2_RANGE[0]

# ── 수치 기울기 ───────────────────────────────────────────────────────────────
def grad_fn(w, eps=1e-4):
    g1 = (cost_fn(w[0]+eps, w[1]) - cost_fn(w[0]-eps, w[1])) / (2*eps)
    g2 = (cost_fn(w[0], w[1]+eps) - cost_fn(w[0], w[1]-eps)) / (2*eps)
    g = np.array([g1, g2])
    norm = np.linalg.norm(g)
    if norm > GRAD_CLIP_NORM:
        g = g * GRAD_CLIP_NORM / norm
    return g

# ── 모멘텀 경사하강법 (2D) ────────────────────────────────────────────────────
def run_gd(w_init):
    w = np.array(w_init, dtype=float)
    v = np.zeros(2)
    hist = [w.copy()]
    w1_lo = W1_RANGE[0] - W1_SPAN * CLIP_OVERSHOOT
    w1_hi = W1_RANGE[1] + W1_SPAN * CLIP_OVERSHOOT
    w2_lo = W2_RANGE[0] - W2_SPAN * CLIP_OVERSHOOT
    w2_hi = W2_RANGE[1] + W2_SPAN * CLIP_OVERSHOOT
    for _ in range(N_EPOCHS):
        g = grad_fn(w)
        v = MOMENTUM * v + g
        w = w - LR * v
        w[0] = np.clip(w[0], w1_lo, w1_hi)
        w[1] = np.clip(w[1], w2_lo, w2_hi)
        hist.append(w.copy())
    return np.array(hist)

# ── 등고선 격자 사전 계산 ─────────────────────────────────────────────────────
w1_vals = np.linspace(W1_RANGE[0], W1_RANGE[1], GRID_RES)
w2_vals = np.linspace(W2_RANGE[0], W2_RANGE[1], GRID_RES)
W1g, W2g = np.meshgrid(w1_vals, w2_vals)
Zg = cost_fn(W1g, W2g)

j_global = Zg.min()
z_span   = Zg.max() - Zg.min()

def normalized_dist(w1a, w2a, w1b, w2b):
    """W1/W2 범위로 정규화된 유클리드 거리. 양축 스케일 차이를 보정."""
    return np.sqrt(((w1a - w1b) / W1_SPAN)**2 + ((w2a - w2b) / W2_SPAN)**2)

def find_global_positions():
    """비용 등고선에서 전역 최솟값 위치를 모두 찾는다."""
    positions = []
    remaining = Zg <= j_global + z_span * GLOBAL_REL_TOL
    while remaining.any():
        idx = np.unravel_index(np.argmin(np.where(remaining, Zg, np.inf)), Zg.shape)
        w = np.array([W1g[idx], W2g[idx]])
        positions.append(w)
        dists = normalized_dist(W1g, W2g, w[0], w[1])
        remaining = remaining & (dists > SPATIAL_REL_TOL)
    return positions

global_positions = find_global_positions()

def is_at_global(w_final):
    return any(
        normalized_dist(w_final[0], w_final[1], gp[0], gp[1]) <= SPATIAL_REL_TOL
        for gp in global_positions
    )

# Phase 1: 단일 시작점 (지역 최솟값 쪽 basin에 위치)
w0 = np.array([W1_RANGE[0] + W1_SPAN * MC_MARGIN_FRAC * 0.5,
               W2_RANGE[1] - W2_SPAN * MC_MARGIN_FRAC])
w_hist_p1 = run_gd(w0)

# Phase 2: 격자 Monte Carlo 시작점 (코너 불안정 구간 제외)
mc_w1_lo = W1_RANGE[0] + W1_SPAN * MC_MARGIN_FRAC
mc_w1_hi = W1_RANGE[1] - W1_SPAN * MC_MARGIN_FRAC
mc_w2_lo = W2_RANGE[0] + W2_SPAN * MC_MARGIN_FRAC
mc_w2_hi = W2_RANGE[1] - W2_SPAN * MC_MARGIN_FRAC
mc_starts = [np.array([w1s, w2s])
             for w1s in np.linspace(mc_w1_lo, mc_w1_hi, MC_GRID_SIZE)
             for w2s in np.linspace(mc_w2_lo, mc_w2_hi, MC_GRID_SIZE)]
mc_hists     = [run_gd(s) for s in mc_starts]
mc_is_global = [is_at_global(h[-1]) for h in mc_hists]

# 3D 궤적용 z(=비용)값 사전 계산 (벡터화된 cost_fn 덕분에 1회 호출)
w_hist_p1_z = cost_fn(w_hist_p1[:, 0], w_hist_p1[:, 1])
mc_hists_z  = [cost_fn(h[:, 0], h[:, 1]) for h in mc_hists]

# ── Figure 레이아웃 ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8.5), facecolor=C_BG)
fig.subplots_adjust(top=0.85, bottom=0.10, left=0.05, right=0.97)

gs_root = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[13, 1.3], hspace=0.45)
# 2D | 3D 사이드바이사이드
gs_main = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_root[0], wspace=0.18)
ax    = fig.add_subplot(gs_main[0])
ax3d  = fig.add_subplot(gs_main[1], projection='3d')
ax.set_facecolor(C_PANEL)

# 버튼 행: [pad][start][restart][next][skip][pad]
gs_btn = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_root[1], wspace=0.35)
ax_b_start   = fig.add_subplot(gs_btn[1])
ax_b_restart = fig.add_subplot(gs_btn[2])
ax_b_next    = fig.add_subplot(gs_btn[3])
ax_b_skip    = fig.add_subplot(gs_btn[4])

# 버튼 axes에 spine/tick 제거 (테두리 없는 평면 버튼)
for btn_ax in [ax_b_start, ax_b_restart, ax_b_next, ax_b_skip]:
    for spine in btn_ax.spines.values():
        spine.set_visible(False)
    btn_ax.set_xticks([])
    btn_ax.set_yticks([])

# 텍스트 전용 버튼: 배경을 figure와 동일하게 두고 hover에서만 미세하게 변함
btn_start   = Button(ax_b_start,   "시작",         color=C_BG, hovercolor=C_HOVER)
btn_restart = Button(ax_b_restart, "재시작",       color=C_BG, hovercolor=C_HOVER)
btn_next    = Button(ax_b_next,    "다음 단계 →",  color=C_BG, hovercolor=C_HOVER)
btn_skip    = Button(ax_b_skip,    "건너뛰기",     color=C_BG, hovercolor=C_HOVER)

for btn, fg, bold, size in [(btn_start,   C_PRIMARY, True,  13),
                            (btn_restart, C_MUTED,   False, 12),
                            (btn_next,    C_PRIMARY, True,  13),
                            (btn_skip,    C_MUTED,   False, 12)]:
    btn.label.set_color(fg)
    btn.label.set_fontsize(size)
    if bold:
        btn.label.set_fontweight('bold')

# ── 2D 등고선 플롯 ────────────────────────────────────────────────────────────
cf = ax.contourf(W1g, W2g, Zg, levels=80, cmap=COST_CMAP, alpha=0.92)
ax.contour(W1g, W2g, Zg, levels=14, colors='#27272a', alpha=0.18, linewidths=0.5)

cbar = plt.colorbar(cf, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label("J(w₁, w₂) — 비용", fontsize=10, color=C_MUTED, labelpad=10)
cbar.ax.tick_params(colors=C_MUTED, labelsize=8)
cbar.outline.set_visible(False)

# 전역 최솟값 마커 (별) — 2D
for i, gp in enumerate(global_positions):
    ax.plot(gp[0], gp[1], '*', color=C_STAR, markersize=24, zorder=6,
            markeredgecolor='white', markeredgewidth=1.8,
            label='전역 최솟값' if i == 0 else '_nolegend_')

leg = ax.legend(fontsize=10, loc='upper right', frameon=True,
                framealpha=0.96, edgecolor='none', facecolor='white',
                borderpad=0.7)
leg.set_zorder(11)

ax.set_xlabel("w₁  (가중치 1)", fontsize=11, color=C_TEXT, labelpad=8)
ax.set_ylabel("w₂  (가중치 2)", fontsize=11, color=C_TEXT, labelpad=8)
ax.set_xlim(W1_RANGE)
ax.set_ylim(W2_RANGE)

# ── 3D Surface 플롯 ───────────────────────────────────────────────────────────
ax3d.plot_surface(W1g, W2g, Zg, cmap=COST_CMAP, alpha=0.85,
                  edgecolor='none', rstride=3, cstride=3, antialiased=True,
                  linewidth=0)

# 3D 축/배경 클린업
ax3d.xaxis.set_pane_color((1, 1, 1, 0))
ax3d.yaxis.set_pane_color((1, 1, 1, 0))
ax3d.zaxis.set_pane_color((1, 1, 1, 0))
ax3d.xaxis._axinfo['grid']['color'] = '#e4e4e7'
ax3d.yaxis._axinfo['grid']['color'] = '#e4e4e7'
ax3d.zaxis._axinfo['grid']['color'] = '#e4e4e7'
ax3d.tick_params(colors=C_MUTED, labelsize=8)
ax3d.set_xlabel("w₁", fontsize=10, color=C_TEXT, labelpad=6)
ax3d.set_ylabel("w₂", fontsize=10, color=C_TEXT, labelpad=6)
ax3d.set_zlabel("J", fontsize=10, color=C_TEXT, labelpad=4)
ax3d.set_xlim(W1_RANGE)
ax3d.set_ylim(W2_RANGE)
ax3d.view_init(elev=28, azim=-58)

# 전역 최솟값 별 (3D) — 표면 약간 위에 띄움
z_offset_star = z_span * 0.04
for gp in global_positions:
    z = float(cost_fn(gp[0], gp[1]))
    ax3d.scatter([gp[0]], [gp[1]], [z + z_offset_star],
                 marker='*', s=260, c=C_STAR,
                 edgecolor='white', linewidths=1.2, zorder=10, depthshade=False)

# 2D Phase 1 동적 요소
trail_p1, = ax.plot([], [], '-', color=C_AMBER, linewidth=2.4, alpha=0.9, zorder=4,
                     solid_capstyle='round', solid_joinstyle='round')
ball_p1,  = ax.plot([], [], 'o', color=C_AMBER, markersize=13, zorder=5,
                     markeredgecolor='white', markeredgewidth=1.8)

# 3D Phase 1 동적 요소
trail_p1_3d, = ax3d.plot([], [], [], '-', color=C_AMBER, linewidth=2.0, alpha=0.95)
ball_p1_3d,  = ax3d.plot([], [], [], 'o', color=C_AMBER, markersize=9,
                          markeredgecolor='white', markeredgewidth=1.2)

# 2D Phase 2 동적 요소
mc_trails = [ax.plot([], [], '-', linewidth=1.3, alpha=0.55, zorder=3,
                      solid_capstyle='round')[0] for _ in mc_starts]
mc_balls  = [ax.plot([], [], 'o', markersize=10, alpha=0.95, zorder=4,
                      markeredgecolor='white', markeredgewidth=1.2)[0] for _ in mc_starts]

# 3D Phase 2 동적 요소
mc_trails_3d = [ax3d.plot([], [], [], '-', linewidth=1.0, alpha=0.7)[0] for _ in mc_starts]
mc_balls_3d  = [ax3d.plot([], [], [], 'o', markersize=7, alpha=0.95,
                           markeredgecolor='white', markeredgewidth=0.8)[0] for _ in mc_starts]

for obj in mc_balls + mc_trails + mc_balls_3d + mc_trails_3d:
    obj.set_visible(False)

# ── 제목 ──────────────────────────────────────────────────────────────────────
main_title = fig.suptitle(
    "경사하강법 시뮬레이터",
    fontsize=18, fontweight='bold', y=0.965, color=C_TEXT
)
english_subtitle = fig.text(
    0.5, 0.918, "Gradient Descent Simulator",
    ha='center', fontsize=10.5, color=C_MUTED, style='italic'
)

ax_subtitle = ax.set_title(
    f"비용 함수 등고선  ·  {DATA_LABEL}",
    fontsize=10, pad=12, color=C_MUTED, loc='left'
)

# 상태 배지 (학습 중 / 도달 / 수렴)
status_txt = ax.text(
    0.5, 0.965, "  대기 중 — 시작 버튼을 눌러주세요  ",
    transform=ax.transAxes, ha='center', va='top',
    fontsize=11, fontweight='bold', color='white',
    bbox=dict(boxstyle='round,pad=0.7', facecolor=C_MUTED, edgecolor='none', alpha=0.95),
    zorder=10
)

# ── 상태 관리 ─────────────────────────────────────────────────────────────────
state = {"phase": 1, "anim": None, "started": False}

def set_status(text, color):
    status_txt.set_text(f"  {text}  ")
    status_txt.get_bbox_patch().set_facecolor(color)

def stop_anim():
    if state["anim"]:
        try:
            state["anim"].event_source.stop()
        except Exception:
            pass
        state["anim"] = None

def set_titles(main_kr, main_en):
    main_title.set_text(main_kr)
    english_subtitle.set_text(main_en)

# ── Phase 종료 처리 ───────────────────────────────────────────────────────────
def finalize_phase1(w_final):
    j_cur = cost_fn(w_final[0], w_final[1])
    is_global = is_at_global(w_final)
    final_color = C_SUCCESS if is_global else C_DANGER
    ball_p1.set_color(final_color)
    ball_p1_3d.set_color(final_color)
    if is_global:
        set_status("목표 지점 도달! (전역 최솟값, Global Min)", C_SUCCESS)
        ax_subtitle.set_text(f"Phase 1 완료 — 전역 최솟값 도달   ·   J = {j_cur:.4f}")
    else:
        set_status("지역 최솟값에 수렴 (Local Min)", C_DANGER)
        ax_subtitle.set_text(f"Phase 1 완료 — 지역 최솟값 수렴   ·   J = {j_cur:.4f}")

def finalize_phase2():
    n_global = int(sum(mc_is_global))
    n_loc = len(mc_starts) - n_global
    set_status(f"완료 — 전역 도달 {n_global}개  /  지역 수렴 {n_loc}개", C_SUCCESS)
    ax_subtitle.set_text(
        f"Phase 2 완료 — 전역 {n_global}개  ·  지역 {n_loc}개   →   몬테카를로가 필요한 이유"
    )

# ── 초기 화면 ─────────────────────────────────────────────────────────────────
def show_initial():
    stop_anim()
    state["phase"] = 1
    state["started"] = False

    for obj in mc_balls + mc_trails + mc_balls_3d + mc_trails_3d:
        obj.set_visible(False)
    for obj in (ball_p1, ball_p1_3d):
        obj.set_visible(False)
    trail_p1.set_data([], [])
    trail_p1.set_visible(False)
    trail_p1_3d.set_data_3d([], [], [])
    trail_p1_3d.set_visible(False)

    set_titles("경사하강법 시뮬레이터", "Gradient Descent Simulator")
    ax_subtitle.set_text(f"비용 함수 등고선  ·  {DATA_LABEL}")
    set_status("대기 중 — 시작 버튼을 눌러주세요", C_MUTED)
    fig.canvas.draw_idle()

# ── Phase 1 ───────────────────────────────────────────────────────────────────
def start_phase1():
    state["phase"] = 1
    state["started"] = True
    stop_anim()

    for obj in mc_balls + mc_trails + mc_balls_3d + mc_trails_3d:
        obj.set_visible(False)
    for obj in (ball_p1, ball_p1_3d):
        obj.set_color(C_AMBER)
        obj.set_visible(True)
    ball_p1.set_data([], [])
    ball_p1_3d.set_data_3d([], [], [])
    trail_p1.set_data([], [])
    trail_p1.set_visible(True)
    trail_p1_3d.set_data_3d([], [], [])
    trail_p1_3d.set_visible(True)

    set_titles("경사하강법 시뮬레이터", "Gradient Descent Simulator")
    ax_subtitle.set_text(f"Phase 1 — 모멘텀 SGD  ·  {DATA_LABEL}")
    set_status("학습 중...  (Training)", C_PRIMARY)

    def update(i):
        # 2D
        trail_p1.set_data(w_hist_p1[:i+1, 0], w_hist_p1[:i+1, 1])
        ball_p1.set_data([w_hist_p1[i, 0]], [w_hist_p1[i, 1]])
        # 3D
        trail_p1_3d.set_data_3d(w_hist_p1[:i+1, 0], w_hist_p1[:i+1, 1],
                                 w_hist_p1_z[:i+1])
        ball_p1_3d.set_data_3d([w_hist_p1[i, 0]], [w_hist_p1[i, 1]],
                                [w_hist_p1_z[i]])
        j_cur = float(w_hist_p1_z[i])
        ax_subtitle.set_text(
            f"Phase 1 — 모멘텀 SGD   ·   에폭 {i}/{N_EPOCHS}   ·   "
            f"w₁={w_hist_p1[i,0]:+.3f}  w₂={w_hist_p1[i,1]:+.3f}   ·   J={j_cur:.4f}"
        )
        if i == N_EPOCHS:
            finalize_phase1(w_hist_p1[i])
        return ball_p1, trail_p1, ball_p1_3d, trail_p1_3d

    state["anim"] = FuncAnimation(fig, update, frames=N_EPOCHS+1,
                                   interval=ANIM_INTERVAL_MS, blit=False, repeat=False)
    fig.canvas.draw_idle()

# ── Phase 2 ───────────────────────────────────────────────────────────────────
def start_phase2():
    state["phase"] = 2
    state["started"] = True
    stop_anim()

    ball_p1.set_visible(False)
    trail_p1.set_visible(False)
    ball_p1_3d.set_visible(False)
    trail_p1_3d.set_visible(False)

    for ball, trail, ball3, trail3, is_glob in zip(
            mc_balls, mc_trails, mc_balls_3d, mc_trails_3d, mc_is_global):
        color = C_SUCCESS if is_glob else C_DANGER
        ball.set_data([], [])
        ball.set_color(color)
        ball.set_visible(True)
        trail.set_color(color)
        trail.set_data([], [])
        trail.set_visible(True)
        ball3.set_data_3d([], [], [])
        ball3.set_color(color)
        ball3.set_visible(True)
        trail3.set_color(color)
        trail3.set_data_3d([], [], [])
        trail3.set_visible(True)

    set_titles("몬테카를로 + 경사하강법 시뮬레이터",
               "Monte Carlo + Gradient Descent Simulator")
    ax_subtitle.set_text(
        f"Phase 2 — 몬테카를로 + 모멘텀 SGD  ·  {len(mc_starts)}개 시작점  "
        f"·  초록=전역  /  빨강=지역"
    )
    set_status("학습 중...  (Monte Carlo)", C_PRIMARY)

    def update(i):
        for ball, trail, ball3, trail3, wh, whz in zip(
                mc_balls, mc_trails, mc_balls_3d, mc_trails_3d, mc_hists, mc_hists_z):
            trail.set_data(wh[:i+1, 0], wh[:i+1, 1])
            ball.set_data([wh[i, 0]], [wh[i, 1]])
            trail3.set_data_3d(wh[:i+1, 0], wh[:i+1, 1], whz[:i+1])
            ball3.set_data_3d([wh[i, 0]], [wh[i, 1]], [whz[i]])
        ax_subtitle.set_text(
            f"Phase 2 — 몬테카를로   ·   에폭 {i}/{N_EPOCHS}   ·   "
            f"초록=전역 최솟값  /  빨강=지역 최솟값"
        )
        if i == N_EPOCHS:
            finalize_phase2()
        return mc_balls + mc_trails + mc_balls_3d + mc_trails_3d

    state["anim"] = FuncAnimation(fig, update, frames=N_EPOCHS+1,
                                   interval=ANIM_INTERVAL_MS, blit=False, repeat=False)
    fig.canvas.draw_idle()

# ── 버튼 콜백 ─────────────────────────────────────────────────────────────────
def on_start(event):
    stop_anim()
    start_phase1()

def on_next(event):
    stop_anim()
    start_phase2()

def on_skip(event):
    if not state["started"]:
        return
    stop_anim()
    if state["phase"] == 1:
        w = w_hist_p1[-1]
        trail_p1.set_data(w_hist_p1[:, 0], w_hist_p1[:, 1])
        ball_p1.set_data([w[0]], [w[1]])
        trail_p1_3d.set_data_3d(w_hist_p1[:, 0], w_hist_p1[:, 1], w_hist_p1_z)
        ball_p1_3d.set_data_3d([w[0]], [w[1]], [w_hist_p1_z[-1]])
        finalize_phase1(w)
    else:
        for ball, trail, ball3, trail3, wh, whz in zip(
                mc_balls, mc_trails, mc_balls_3d, mc_trails_3d, mc_hists, mc_hists_z):
            trail.set_data(wh[:, 0], wh[:, 1])
            ball.set_data([wh[-1, 0]], [wh[-1, 1]])
            trail3.set_data_3d(wh[:, 0], wh[:, 1], whz)
            ball3.set_data_3d([wh[-1, 0]], [wh[-1, 1]], [whz[-1]])
        finalize_phase2()
    fig.canvas.draw_idle()

btn_start.on_clicked(on_start)
btn_restart.on_clicked(on_start)
btn_next.on_clicked(on_next)
btn_skip.on_clicked(on_skip)

show_initial()
plt.show()
