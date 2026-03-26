"""
알고리즘 동작 원리 애니메이션

[상단 행] Monte Carlo 샘플링 과정
  - 왼쪽: N(0, σ²) 분포에서 ε 샘플링 (떨어지는 화살표)
  - 오른쪽: y_true 선 위에 노이즈가 더해져 데이터 포인트가 쌓이는 과정

[하단 행] 경사하강법 스텝 과정
  - 왼쪽: J(w) 곡선 위 현재 위치 + 접선(기울기) + 스텝 화살표
  - 오른쪽: 산점도 위 직선이 스텝마다 이동하는 과정
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from generate import generate_dataset
from gradient_descent import gradient_descent, cost, gradients

# ── 설정 ──────────────────────────────────────────────────────────────────────
W_TRUE       = 3.0
B_TRUE       = 7.0
NOISE_STD    = 5.0
N_SAMPLES    = 80          # MC 패널용 (천천히 쌓이도록 적게)
LR           = 0.05
N_EPOCHS_GD  = 120         # GD 패널 스텝 수
SEED         = 42

# ── 데이터 & 학습 ──────────────────────────────────────────────────────────────
x_all, y_all = generate_dataset(N_SAMPLES, W_TRUE, B_TRUE, NOISE_STD, seed=SEED)

rng = np.random.default_rng(SEED)
w_init = rng.uniform(0.0, 1.5)
b_init = rng.uniform(0.0, 3.0)

result = gradient_descent(x_all, y_all, LR, N_EPOCHS_GD, w_init, b_init)
w_hist = result["w_history"]
b_hist = result["b_history"]

# ── J(w) 곡선: b를 b_true에 고정해 1D 슬라이스 ──────────────────────────────
B_FIXED  = B_TRUE
w_grid   = np.linspace(-1.0, 7.0, 300)
j_grid   = np.array([(1 / (2 * N_SAMPLES)) * np.sum((w * x_all + B_FIXED - y_all) ** 2)
                     for w in w_grid])

# ── Figure ───────────────────────────────────────────────────────────────────
BG_COL   = "#0d0d1f"
AX_COL   = "#12122a"
TXT_COL  = "#ddddff"
GRID_COL = "#333355"

fig = plt.figure(figsize=(13, 8), facecolor=BG_COL)
fig.patch.set_facecolor(BG_COL)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.35)

ax_bell  = fig.add_subplot(gs[0, 0])   # MC: 분포
ax_data  = fig.add_subplot(gs[0, 1])   # MC: 데이터 생성
ax_jw    = fig.add_subplot(gs[1, 0])   # GD: J(w) 곡선
ax_line  = fig.add_subplot(gs[1, 1])   # GD: 회귀선 이동

for ax in [ax_bell, ax_data, ax_jw, ax_line]:
    ax.set_facecolor(AX_COL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.tick_params(colors="#9999bb", labelsize=8)
    ax.xaxis.label.set_color(TXT_COL)
    ax.yaxis.label.set_color(TXT_COL)
    ax.title.set_color(TXT_COL)
    ax.grid(color=GRID_COL, linewidth=0.4, linestyle="--", alpha=0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# [상단 왼쪽] Monte Carlo: 가우시안 분포 패널
# ═══════════════════════════════════════════════════════════════════════════════
eps_range = np.linspace(-3 * NOISE_STD, 3 * NOISE_STD, 400)
bell_y    = (1 / (NOISE_STD * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (eps_range / NOISE_STD) ** 2)

ax_bell.plot(eps_range, bell_y, color="#a78bfa", linewidth=2, label=r"$\mathcal{N}(0,\sigma^2)$")
ax_bell.fill_between(eps_range, bell_y, alpha=0.15, color="#a78bfa")
ax_bell.set_xlim(eps_range[0], eps_range[-1])
ax_bell.set_ylim(-0.012, bell_y.max() * 1.25)
ax_bell.set_xlabel("ε  (noise)", fontsize=9)
ax_bell.set_ylabel("Density", fontsize=9)
ax_bell.set_title("Step 1 · Monte Carlo Sampling\n"
                  r"$\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$", fontsize=9)

# 동적: 샘플된 ε 위치 마커 + 드롭 화살표
bell_vline   = ax_bell.axvline(0, color="cyan", linewidth=1.5, linestyle="--", alpha=0)
bell_dot,    = ax_bell.plot([], [], "o", color="cyan", markersize=9, zorder=6)
bell_arr     = ax_bell.annotate("", xy=(0, 0), xytext=(0, 0),
                                arrowprops=dict(arrowstyle="-|>",
                                                color="cyan", lw=1.8))
# ε 값 텍스트
eps_text     = ax_bell.text(0.05, 0.88, "", transform=ax_bell.transAxes,
                            color="cyan", fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════════
# [상단 오른쪽] Monte Carlo: 데이터 포인트 생성 패널
# ═══════════════════════════════════════════════════════════════════════════════
x_line_plt = np.linspace(x_all.min() - 0.3, x_all.max() + 0.3, 200)

ax_data.plot(x_line_plt, W_TRUE * x_line_plt + B_TRUE,
             color="#4ade80", linewidth=2, zorder=3, label=r"$y_{true} = wx + b$")
ax_data.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
ax_data.set_ylim(y_all.min() - 3, y_all.max() + 3)
ax_data.set_xlabel("x", fontsize=9)
ax_data.set_ylabel("y", fontsize=9)
ax_data.set_title("Step 2 · Data Point Generation\n"
                  r"$y_i = wx_i + b + \varepsilon_i$", fontsize=9)
ax_data.legend(fontsize=8, facecolor="#1a1a33", labelcolor=TXT_COL, loc="upper left")

data_scatter = ax_data.scatter([], [], s=25, color="steelblue", zorder=4, alpha=0.8)
# 현재 포인트 강조 + 수직 오차선
cur_dot,     = ax_data.plot([], [], "o", color="cyan", markersize=9, zorder=7)
noise_line,  = ax_data.plot([], [], color="cyan", linewidth=1.5, linestyle=":", zorder=6)
true_dot,    = ax_data.plot([], [], "o", color="#4ade80", markersize=7, zorder=6)
data_text    = ax_data.text(0.03, 0.07, "", transform=ax_data.transAxes,
                            color="cyan", fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════════
# [하단 왼쪽] 경사하강법: J(w) 곡선 + 접선 + 스텝
# ═══════════════════════════════════════════════════════════════════════════════
ax_jw.plot(w_grid, j_grid, color="#f97316", linewidth=2.5, zorder=2)
ax_jw.axvline(W_TRUE, color="#4ade80", linewidth=1, linestyle="--", alpha=0.7,
              label=f"w_true = {W_TRUE}")
ax_jw.set_xlim(w_grid[0], w_grid[-1])
ax_jw.set_ylim(j_grid.min() * 0.8, j_grid.max() * 0.25)
ax_jw.set_xlabel("w", fontsize=9)
ax_jw.set_ylabel("J(w)", fontsize=9)
ax_jw.set_title("Step 3 · Gradient Descent\n"
                r"$w \leftarrow w - \alpha \cdot \partial J/\partial w$", fontsize=9)
ax_jw.legend(fontsize=8, facecolor="#1a1a33", labelcolor=TXT_COL)

cur_pt,     = ax_jw.plot([], [], "o", color="white", markersize=9, zorder=6)
tangent_ln, = ax_jw.plot([], [], color="#facc15", linewidth=2, alpha=0.85, zorder=5)
step_arr    = ax_jw.annotate("", xy=(0, 0), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="-|>", color="cyan",
                                             lw=2.0, mutation_scale=14))
gd_text     = ax_jw.text(0.03, 0.88, "", transform=ax_jw.transAxes,
                         color="#facc15", fontsize=8.5)

# ═══════════════════════════════════════════════════════════════════════════════
# [하단 오른쪽] 경사하강법: 회귀선 이동
# ═══════════════════════════════════════════════════════════════════════════════
ax_line.scatter(x_all, y_all, s=15, alpha=0.4, color="steelblue", zorder=2)
ax_line.plot(x_line_plt, W_TRUE * x_line_plt + B_TRUE,
             color="#4ade80", linewidth=1.5, linestyle="--",
             alpha=0.7, label=f"True (w={W_TRUE}, b={B_TRUE})")
ax_line.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
ax_line.set_ylim(y_all.min() - 3, y_all.max() + 3)
ax_line.set_xlabel("x", fontsize=9)
ax_line.set_ylabel("y", fontsize=9)
ax_line.set_title("Step 4 · Line Converging", fontsize=9)
ax_line.legend(fontsize=8, facecolor="#1a1a33", labelcolor=TXT_COL)

reg_ln,  = ax_line.plot([], [], color="orangered", linewidth=2.5, zorder=4)
reg_txt  = ax_line.text(0.03, 0.07, "", transform=ax_line.transAxes,
                        color="orangered", fontsize=8)

# 에포크 표시
epoch_txt = fig.text(0.5, 0.98, "", ha="center", va="top",
                     color="white", fontsize=11, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# 프레임 설계
# ═══════════════════════════════════════════════════════════════════════════════
# Phase A: MC 샘플링 (N_SAMPLES 프레임)
# Phase B: GD 스텝  (N_EPOCHS_GD 프레임)
PHASE_A = N_SAMPLES
PHASE_B = N_EPOCHS_GD
TOTAL   = PHASE_A + PHASE_B

# 노이즈 값 사전 계산 (y_all - true값)
eps_all = y_all - (W_TRUE * x_all + B_TRUE)

# 누적 scatter 데이터
collected_x = []
collected_y = []

def update(frame):
    global collected_x, collected_y

    # ── Phase A: Monte Carlo ──────────────────────────────────────────────────
    if frame < PHASE_A:
        i      = frame
        xi     = x_all[i]
        yi     = y_all[i]
        eps_i  = eps_all[i]
        y_true = W_TRUE * xi + B_TRUE

        epoch_txt.set_text(f"Phase 1 · Monte Carlo Sampling  [{i+1}/{PHASE_A}]")

        # 종 모양: 샘플 위치 강조
        bell_vline.set_xdata([eps_i, eps_i])
        bell_vline.set_alpha(0.6)
        bell_pdf_val = (1 / (NOISE_STD * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (eps_i / NOISE_STD) ** 2)
        bell_dot.set_data([eps_i], [bell_pdf_val])

        # 드롭 화살표 (분포 위 → x축)
        bell_arr.xy     = (eps_i, 0)
        bell_arr.xytext = (eps_i, bell_pdf_val * 0.6)
        bell_arr.set_visible(True)
        eps_text.set_text(f"ε = {eps_i:+.2f}")

        # 데이터 패널: 누적 + 현재 포인트
        collected_x.append(xi)
        collected_y.append(yi)
        data_scatter.set_offsets(np.c_[collected_x, collected_y])

        cur_dot.set_data([xi], [yi])
        true_dot.set_data([xi], [y_true])
        noise_line.set_data([xi, xi], [y_true, yi])   # 수직 노이즈 선
        data_text.set_text(f"x={xi:.1f}  y_true={y_true:.1f}  ε={eps_i:+.1f}  y={yi:.1f}")

        # GD 패널: 초기 상태 고정
        w0 = w_hist[0]
        j0 = (1 / (2 * N_SAMPLES)) * np.sum((w0 * x_all + B_FIXED - y_all) ** 2)
        cur_pt.set_data([w0], [j0])
        tangent_ln.set_data([], [])
        step_arr.xy = step_arr.xytext = (w0, j0)
        gd_text.set_text("")
        reg_ln.set_data(x_line_plt, w0 * x_line_plt + b_hist[0])
        reg_txt.set_text(f"w={w0:.3f}  b={b_hist[0]:.3f}")

    # ── Phase B: Gradient Descent ─────────────────────────────────────────────
    else:
        gd_step = frame - PHASE_A      # 0 ~ N_EPOCHS_GD-1
        epoch_txt.set_text(f"Phase 2 · Gradient Descent  Epoch {gd_step+1}/{PHASE_B}"
                           f"  |  w={w_hist[gd_step]:.3f}  b={b_hist[gd_step]:.3f}")

        # MC 패널: 정적 유지 (마지막 상태)
        bell_vline.set_alpha(0.2)
        cur_dot.set_data([], [])
        noise_line.set_data([], [])
        true_dot.set_data([], [])
        data_text.set_text("")
        data_scatter.set_offsets(np.c_[x_all, y_all])

        # J(w) 패널: 현재 w, 접선, 스텝 화살표
        w_cur = w_hist[gd_step]
        j_cur = (1 / (2 * N_SAMPLES)) * np.sum((w_cur * x_all + B_FIXED - y_all) ** 2)

        # 접선: dJ/dw 수치 미분
        dj_dw = (1 / N_SAMPLES) * np.sum((w_cur * x_all + B_FIXED - y_all) * x_all)
        tan_w = np.array([w_cur - 0.8, w_cur + 0.8])
        tan_j = j_cur + dj_dw * (tan_w - w_cur)
        tangent_ln.set_data(tan_w, tan_j)

        cur_pt.set_data([w_cur], [j_cur])

        # 스텝 화살표: 현재 w → 다음 w
        w_next = w_hist[gd_step + 1]
        j_next = (1 / (2 * N_SAMPLES)) * np.sum((w_next * x_all + B_FIXED - y_all) ** 2)
        step_arr.xy     = (w_next, j_next)
        step_arr.xytext = (w_cur,  j_cur)
        step_arr.set_visible(True)

        gd_text.set_text(f"∂J/∂w = {dj_dw:.3f}\n"
                         f"step = -α·(∂J/∂w) = {-LR * dj_dw:.3f}")

        # 회귀선
        reg_ln.set_data(x_line_plt, w_cur * x_line_plt + b_hist[gd_step])
        reg_txt.set_text(f"w={w_cur:.3f}  b={b_hist[gd_step]:.3f}")

    return (bell_vline, bell_dot, eps_text,
            data_scatter, cur_dot, noise_line, true_dot, data_text,
            cur_pt, tangent_ln, step_arr, gd_text,
            reg_ln, reg_txt, epoch_txt)


anim = FuncAnimation(fig, update,
                     frames=TOTAL,
                     interval=80,
                     blit=False)

print("GIF 저장 중...")
anim.save("explain.gif", writer=PillowWriter(fps=15), dpi=110)
print("Saved: explain.gif")

plt.show()
