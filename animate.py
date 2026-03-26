"""
경사하강법 애니메이션
- 왼쪽: J(w,b) 3D 비용 곡면 위를 내려오는 경로
- 오른쪽: 2D 등고선 + 이동 궤적
- 하단: 산점도 위 직선 수렴 과정
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

from generate import generate_dataset
from gradient_descent import gradient_descent

# ── 설정 ──────────────────────────────────────────────────────────────────────
W_TRUE        = 3.0
B_TRUE        = 7.0
NOISE_STD     = 5.0
N_SAMPLES     = 200
LEARNING_RATE = 0.05
N_EPOCHS      = 300      # 애니메이션 프레임 수 = N_EPOCHS
SEED          = 42
INTERVAL_MS   = 30       # 프레임 간격 (ms)
N_FRAMES      = 150      # 실제 렌더링 프레임 수 (epoch을 이만큼 균등 샘플)

# ── 데이터 생성 & 학습 전체 실행 ──────────────────────────────────────────────
x, y = generate_dataset(N_SAMPLES, W_TRUE, B_TRUE, NOISE_STD, seed=SEED)

rng = np.random.default_rng(SEED)
w_init = rng.uniform(-2.0, 2.0)
b_init = rng.uniform(-2.0, 2.0)

result = gradient_descent(x, y,
                          learning_rate=LEARNING_RATE,
                          n_epochs=N_EPOCHS,
                          w_init=w_init,
                          b_init=b_init)

w_hist = result["w_history"]   # shape: (N_EPOCHS+1,)
b_hist = result["b_history"]
cost_hist = result["cost_history"]

# 애니메이션용 프레임 인덱스 (균등 샘플)
frame_indices = np.unique(np.linspace(0, N_EPOCHS, N_FRAMES, dtype=int))
N_FRAMES = len(frame_indices)

# ── J(w,b) 곡면 사전 계산 ─────────────────────────────────────────────────────
w_margin = max(abs(w_init - W_TRUE), 2.0) * 1.5
b_margin = max(abs(b_init - B_TRUE), 3.0) * 1.5

w_range = np.linspace(W_TRUE - w_margin, W_TRUE + w_margin, 80)
b_range = np.linspace(B_TRUE - b_margin, B_TRUE + b_margin, 80)
WG, BG = np.meshgrid(w_range, b_range)

# 벡터화된 비용 계산
JG = np.zeros_like(WG)
N = len(x)
for i in range(WG.shape[0]):
    for j in range(WG.shape[1]):
        pred = WG[i, j] * x + BG[i, j]
        JG[i, j] = (1 / (2 * N)) * np.sum((pred - y) ** 2)

# ── Figure 레이아웃 ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor="#0e0e1a")
fig.patch.set_facecolor("#0e0e1a")

gs = gridspec.GridSpec(2, 2, figure=fig,
                       height_ratios=[1.4, 1],
                       hspace=0.4, wspace=0.3)

ax3d      = fig.add_subplot(gs[0, 0], projection="3d")
ax_cont   = fig.add_subplot(gs[0, 1])
ax_line   = fig.add_subplot(gs[1, 0])
ax_cost   = fig.add_subplot(gs[1, 1])

for ax in [ax_cont, ax_line, ax_cost]:
    ax.set_facecolor("#12122a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("#ddddff")

ax3d.set_facecolor("#0e0e1a")

# ── 3D 곡면 (정적) ───────────────────────────────────────────────────────────
surf = ax3d.plot_surface(WG, BG, JG,
                         cmap="plasma", alpha=0.55,
                         linewidth=0, antialiased=True)
ax3d.contour(WG, BG, JG, zdir="z",
             offset=JG.min() - 2, levels=15, cmap="plasma", alpha=0.4)
ax3d.set_xlabel("w", color="#aaaacc", labelpad=4, fontsize=8)
ax3d.set_ylabel("b", color="#aaaacc", labelpad=4, fontsize=8)
ax3d.set_zlabel("J(w,b)", color="#aaaacc", labelpad=4, fontsize=8)
ax3d.set_title("Cost Surface", color="#ddddff", fontsize=10, pad=6)
ax3d.tick_params(colors="#777799", labelsize=7)
ax3d.view_init(elev=28, azim=-60)

# 최솟값 마커
ax3d.scatter([W_TRUE], [B_TRUE], [JG.min()],
             color="lime", s=60, zorder=10)

# 3D 경로 (동적)
line3d, = ax3d.plot([], [], [], "o-",
                    color="cyan", markersize=3, linewidth=1.5, alpha=0.9)
dot3d,  = ax3d.plot([], [], [], "o",
                    color="white", markersize=7, zorder=10)

# ── 2D 등고선 (정적) ─────────────────────────────────────────────────────────
levels = np.percentile(JG, np.linspace(2, 98, 25))
ax_cont.contourf(WG, BG, JG, levels=levels, cmap="plasma", alpha=0.7)
ax_cont.contour(WG,  BG, JG, levels=levels, colors="white", linewidths=0.4, alpha=0.3)
ax_cont.scatter([W_TRUE], [B_TRUE], color="lime", s=80, zorder=5, label="True minimum")
ax_cont.set_xlabel("w", fontsize=9)
ax_cont.set_ylabel("b", fontsize=9)
ax_cont.set_title("Contour + GD Path", fontsize=10)

line_cont, = ax_cont.plot([], [], "o-",
                          color="cyan", markersize=2, linewidth=1.5, alpha=0.85)
dot_cont,  = ax_cont.plot([], [], "o",
                          color="white", markersize=7, zorder=6)

# ── 산점도 (정적) ────────────────────────────────────────────────────────────
ax_line.scatter(x, y, s=12, alpha=0.4, color="steelblue")
ax_line.plot(np.sort(x), W_TRUE * np.sort(x) + B_TRUE,
             "g--", linewidth=1.5, alpha=0.7, label=f"True (w={W_TRUE}, b={B_TRUE})")
reg_line, = ax_line.plot([], [], color="orangered", linewidth=2.5, label="Learned")
ax_line.set_xlim(x.min() - 0.5, x.max() + 0.5)
ax_line.set_ylim(y.min() - 2, y.max() + 2)
ax_line.set_xlabel("x", fontsize=9)
ax_line.set_ylabel("y", fontsize=9)
ax_line.set_title("Regression Line", fontsize=10)
ax_line.legend(fontsize=7, facecolor="#1a1a33", labelcolor="#ccccee")

# ── 비용 곡선 (동적) ────────────────────────────────────────────────────────
ax_cost.set_xlim(0, N_EPOCHS)
ax_cost.set_ylim(cost_hist[-1] * 0.95, cost_hist[0] * 1.05)
ax_cost.set_yscale("log")
ax_cost.axhline((NOISE_STD ** 2) / 2, color="lime", linestyle="--",
                linewidth=1, alpha=0.7, label="Noise floor (σ²/2)")
ax_cost.set_xlabel("Epoch", fontsize=9)
ax_cost.set_ylabel("J(w,b)", fontsize=9)
ax_cost.set_title("Cost Curve", fontsize=10)
ax_cost.legend(fontsize=7, facecolor="#1a1a33", labelcolor="#ccccee")

cost_line, = ax_cost.plot([], [], color="darkorange", linewidth=1.8)

epoch_text = fig.text(0.5, 0.97,
                      "Epoch: 0", ha="center", va="top",
                      color="#ffffff", fontsize=11,
                      fontweight="bold")

# ── J(w,b) 경로 값 사전 계산 ──────────────────────────────────────────────────
def j_at(w, b):
    return (1 / (2 * N)) * np.sum((w * x + b - y) ** 2)

j_path = np.array([j_at(w_hist[i], b_hist[i]) for i in range(len(w_hist))])

# ── 애니메이션 함수 ───────────────────────────────────────────────────────────
def update(frame_num):
    idx = frame_indices[frame_num]         # 실제 epoch 인덱스

    ws = w_hist[:idx + 1]
    bs = b_hist[:idx + 1]
    js = j_path[:idx + 1]

    # 3D
    line3d.set_data(ws, bs)
    line3d.set_3d_properties(js)
    dot3d.set_data([ws[-1]], [bs[-1]])
    dot3d.set_3d_properties([js[-1]])

    # 등고선
    line_cont.set_data(ws, bs)
    dot_cont.set_data([ws[-1]], [bs[-1]])

    # 회귀선
    x_sorted = np.array([x.min(), x.max()])
    reg_line.set_data(x_sorted, ws[-1] * x_sorted + bs[-1])

    # 비용 곡선
    cost_line.set_data(np.arange(len(cost_hist[:idx])), cost_hist[:idx])

    epoch_text.set_text(
        f"Epoch: {idx:4d}  |  w={ws[-1]:.3f}  b={bs[-1]:.3f}  J={js[-1]:.3f}"
    )

    return line3d, dot3d, line_cont, dot_cont, reg_line, cost_line, epoch_text


anim = FuncAnimation(fig, update,
                     frames=N_FRAMES,
                     interval=INTERVAL_MS,
                     blit=False)

print("GIF 저장 중... (잠시 기다려 주세요)")
anim.save("gradient_descent.gif",
          writer=PillowWriter(fps=30),
          dpi=110)
print("Saved: gradient_descent.gif")

plt.show()
