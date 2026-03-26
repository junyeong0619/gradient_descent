"""
GD vs MC-GD Interactive Simulator
- Himmelblau function: f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
- 4 local minima -> GD gets trapped, MC-GD can escape
- Controls: Run GD / Run MC-GD / Reset buttons, lr & sample sliders, mouse click
"""

import matplotlib
matplotlib.use("macosx")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation

# ── Himmelblau 함수 & 그래디언트 ────────────────────────────────────────────
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_grad(x, y):
    dfdx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dfdy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return dfdx, dfdy

# 4개의 글로벌 미니멈
MINIMA = [(3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]

# ── 알고리즘 ─────────────────────────────────────────────────────────────────
def run_gd(x0, y0, lr, n_steps=300):
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(n_steps):
        gx, gy = himmelblau_grad(x, y)
        x = x - lr * gx
        y = y - lr * gy
        # 범위 클리핑
        x = np.clip(x, -5, 5)
        y = np.clip(y, -5, 5)
        path.append((x, y))
        if gx**2 + gy**2 < 1e-10:
            break
    return np.array(path)

def run_mc_gd(x0, y0, lr, n_samples, n_steps=300):
    """
    현재 위치 주변 n_samples개 랜덤 포인트 샘플링 후
    가장 낮은 손실 방향으로 이동 (gradient 미사용)
    """
    path = [(x0, y0)]
    x, y = x0, y0
    rng = np.random.default_rng()
    for _ in range(n_steps):
        # 현재 반경 lr*10 범위에서 무작위 탐색
        radius = lr * 15
        dx = rng.uniform(-radius, radius, n_samples)
        dy = rng.uniform(-radius, radius, n_samples)
        candidates_x = np.clip(x + dx, -5, 5)
        candidates_y = np.clip(y + dy, -5, 5)
        losses = himmelblau(candidates_x, candidates_y)
        best = np.argmin(losses)
        # 현재보다 나으면 이동
        if losses[best] < himmelblau(x, y):
            x = candidates_x[best]
            y = candidates_y[best]
        path.append((x, y))
    return np.array(path)

# ── 등고선 사전 계산 ──────────────────────────────────────────────────────────
RANGE = 5.0
res = 300
xs = np.linspace(-RANGE, RANGE, res)
ys = np.linspace(-RANGE, RANGE, res)
XG, YG = np.meshgrid(xs, ys)
ZG = himmelblau(XG, YG)
LEVELS = np.unique(np.concatenate([
    np.linspace(0.1, 10, 15),
    np.linspace(11, 100, 10),
    np.linspace(110, 500, 5),
]))

# ── 상태 ─────────────────────────────────────────────────────────────────────
state = {
    "start": (-4.0, -1.0),   # 시작점
    "gd_path": None,
    "mc_path": None,
    "anim": None,
    "frame": 0,
}

# ── Figure 레이아웃 ───────────────────────────────────────────────────────────
BG    = "#0e0e1a"
AX    = "#12122a"
TXT   = "#ddddff"
GRID  = "#333355"

fig = plt.figure(figsize=(14, 8), facecolor=BG)
fig.patch.set_facecolor(BG)

# 메인 영역 / 컨트롤 영역
gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                             height_ratios=[10, 1.4],
                             hspace=0.05)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 2,
                                          subplot_spec=gs_outer[0],
                                          width_ratios=[1.5, 1],
                                          wspace=0.3)
gs_right = gridspec.GridSpecFromSubplotSpec(2, 1,
                                            subplot_spec=gs_top[1],
                                            hspace=0.45)

ax_cont  = fig.add_subplot(gs_top[0])        # 등고선 + 경로
ax_loss  = fig.add_subplot(gs_right[0])      # 손실 그래프
ax_info  = fig.add_subplot(gs_right[1])      # 정보 텍스트

for ax in [ax_cont, ax_loss]:
    ax.set_facecolor(AX)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors="#9999bb", labelsize=8)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(TXT)
    ax.grid(color=GRID, linewidth=0.4, linestyle="--", alpha=0.5)

ax_info.set_facecolor(AX)
ax_info.axis("off")
for sp in ax_info.spines.values():
    sp.set_edgecolor(GRID)

# ── 등고선 (정적) ─────────────────────────────────────────────────────────────
ax_cont.contourf(XG, YG, ZG, levels=LEVELS, cmap="plasma", alpha=0.75)
ax_cont.contour(XG,  YG, ZG, levels=LEVELS, colors="white",
                linewidths=0.3, alpha=0.25)
for mx, my in MINIMA:
    ax_cont.plot(mx, my, "*", color="lime", markersize=10, zorder=5)
ax_cont.set_xlim(-RANGE, RANGE)
ax_cont.set_ylim(-RANGE, RANGE)
ax_cont.set_xlabel("x", fontsize=9)
ax_cont.set_ylabel("y", fontsize=9)
ax_cont.set_title("Himmelblau  f(x,y) = (x²+y−11)² + (x+y²−7)²\n"
                  "★ = local minima (x4)",
                  fontsize=9, color=TXT)

# 시작점 마커
start_dot, = ax_cont.plot(*state["start"], "o",
                           color="white", markersize=9, zorder=10)

# GD 경로
gd_line,  = ax_cont.plot([], [], "-", color="cyan",
                          linewidth=1.5, alpha=0.85, label="GD")
gd_dot,   = ax_cont.plot([], [], "o", color="cyan", markersize=6, zorder=8)

# MC-GD 경로
mc_line,  = ax_cont.plot([], [], "-", color="orange",
                          linewidth=1.5, alpha=0.85, label="MC-GD")
mc_dot,   = ax_cont.plot([], [], "o", color="orange", markersize=6, zorder=8)

ax_cont.legend(fontsize=8, facecolor="#1a1a33", labelcolor=TXT,
               loc="upper right")

# ── 손실 그래프 (동적) ────────────────────────────────────────────────────────
ax_loss.set_title("Loss  f(x, y)", fontsize=9)
ax_loss.set_xlabel("Step", fontsize=8)
ax_loss.set_ylabel("f(x,y)", fontsize=8)
ax_loss.set_yscale("log")

gd_loss_line, = ax_loss.plot([], [], color="cyan",   linewidth=1.5, label="GD")
mc_loss_line, = ax_loss.plot([], [], color="orange", linewidth=1.5, label="MC-GD")
ax_loss.legend(fontsize=7, facecolor="#1a1a33", labelcolor=TXT)

# ── 정보 텍스트 ───────────────────────────────────────────────────────────────
info_text = ax_info.text(0.05, 0.95, "Click on the contour to set start point, then press Run.",
                         transform=ax_info.transAxes,
                         va="top", ha="left", fontsize=8.5,
                         color=TXT, linespacing=1.6)

# ── 컨트롤 영역 ───────────────────────────────────────────────────────────────
ctrl_gs = gridspec.GridSpecFromSubplotSpec(1, 7,
                                           subplot_spec=gs_outer[1],
                                           wspace=0.5)

ax_btn_gd    = fig.add_subplot(ctrl_gs[0])
ax_btn_mc    = fig.add_subplot(ctrl_gs[1])
ax_btn_reset = fig.add_subplot(ctrl_gs[2])
ax_sl_lr     = fig.add_subplot(ctrl_gs[3:5])
ax_sl_n      = fig.add_subplot(ctrl_gs[5:7])

btn_gd    = Button(ax_btn_gd,    "Run GD",    color="#1a1a3a", hovercolor="#2a2a5a")
btn_mc    = Button(ax_btn_mc,    "Run MC-GD", color="#1a1a3a", hovercolor="#2a2a5a")
btn_reset = Button(ax_btn_reset, "Reset",     color="#1a1a3a", hovercolor="#3a1a1a")

for btn in [btn_gd, btn_mc, btn_reset]:
    btn.label.set_color(TXT)
    btn.label.set_fontsize(9)

sl_lr = Slider(ax_sl_lr, "lr", 0.001, 0.15,
               valinit=0.01, color="#3355aa",
               track_color="#222244")
sl_lr.label.set_color(TXT)
sl_lr.valtext.set_color(TXT)

sl_n = Slider(ax_sl_n, "MC samples", 10, 300,
              valinit=50, valstep=10, color="#aa5533",
              track_color="#332222")
sl_n.label.set_color(TXT)
sl_n.valtext.set_color(TXT)

# ── 애니메이션 헬퍼 ───────────────────────────────────────────────────────────
def stop_anim():
    if state["anim"] is not None:
        try:
            state["anim"].event_source.stop()
        except Exception:
            pass
        state["anim"] = None

def update_info(step, gd_path, mc_path):
    lines = []
    sx, sy = state["start"]
    lines.append(f"Start : ({sx:.2f}, {sy:.2f})")
    lines.append(f"Step  : {step}")

    if gd_path is not None and step < len(gd_path):
        gx, gy = gd_path[step]
        lines.append(f"GD    : ({gx:.3f}, {gy:.3f})  f={himmelblau(gx,gy):.4f}")
    if mc_path is not None and step < len(mc_path):
        mx, my = mc_path[step]
        lines.append(f"MC-GD : ({mx:.3f}, {my:.3f})  f={himmelblau(mx,my):.4f}")
    info_text.set_text("\n".join(lines))

def animate_paths(gd_path, mc_path):
    stop_anim()
    n_frames = max(
        len(gd_path) if gd_path is not None else 0,
        len(mc_path) if mc_path is not None else 0,
    )

    # 손실값 미리 계산
    gd_losses = (himmelblau(gd_path[:, 0], gd_path[:, 1])
                 if gd_path is not None else None)
    mc_losses = (himmelblau(mc_path[:, 0], mc_path[:, 1])
                 if mc_path is not None else None)

    # 손실 그래프 초기화 및 축 범위 설정
    gd_loss_line.set_data([], [])
    mc_loss_line.set_data([], [])
    all_losses = []
    if gd_losses is not None: all_losses.append(gd_losses)
    if mc_losses is not None: all_losses.append(mc_losses)
    all_losses = np.concatenate(all_losses)
    ax_loss.set_xlim(0, n_frames)
    ymin = max(all_losses.min() * 0.5, 1e-6)
    ax_loss.set_ylim(ymin, all_losses.max() * 2)

    def update(frame):
        gd_idx = min(frame, len(gd_path) - 1) if gd_path is not None else None
        mc_idx = min(frame, len(mc_path) - 1) if mc_path is not None else None

        if gd_idx is not None:
            gd_line.set_data(gd_path[:gd_idx+1, 0], gd_path[:gd_idx+1, 1])
            gd_dot.set_data([gd_path[gd_idx, 0]], [gd_path[gd_idx, 1]])
            gd_loss_line.set_data(np.arange(gd_idx+1), gd_losses[:gd_idx+1])

        if mc_idx is not None:
            mc_line.set_data(mc_path[:mc_idx+1, 0], mc_path[:mc_idx+1, 1])
            mc_dot.set_data([mc_path[mc_idx, 0]], [mc_path[mc_idx, 1]])
            mc_loss_line.set_data(np.arange(mc_idx+1), mc_losses[:mc_idx+1])

        update_info(frame, gd_path, mc_path)
        return gd_line, gd_dot, mc_line, mc_dot, gd_loss_line, mc_loss_line, info_text

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=30, blit=False, repeat=False)
    state["anim"] = anim
    fig.canvas.draw_idle()

# ── 버튼 콜백 ─────────────────────────────────────────────────────────────────
def on_run_gd(event):
    stop_anim()
    sx, sy = state["start"]
    lr = sl_lr.val
    path = run_gd(sx, sy, lr)
    state["gd_path"] = path
    # MC 경로는 유지
    animate_paths(state["gd_path"], state["mc_path"])

def on_run_mc(event):
    stop_anim()
    sx, sy = state["start"]
    lr = sl_lr.val
    n  = int(sl_n.val)
    path = run_mc_gd(sx, sy, lr, n)
    state["mc_path"] = path
    animate_paths(state["gd_path"], state["mc_path"])

def on_reset(event):
    stop_anim()
    state["gd_path"] = None
    state["mc_path"] = None
    gd_line.set_data([], [])
    gd_dot.set_data([], [])
    mc_line.set_data([], [])
    mc_dot.set_data([], [])
    gd_loss_line.set_data([], [])
    mc_loss_line.set_data([], [])
    info_text.set_text("Click on the contour to set start point, then press Run.")
    fig.canvas.draw_idle()

btn_gd.on_clicked(on_run_gd)
btn_mc.on_clicked(on_run_mc)
btn_reset.on_clicked(on_reset)

# ── 마우스 클릭 → 시작점 변경 ────────────────────────────────────────────────
def on_click(event):
    if event.inaxes is not ax_cont:
        return
    if event.button != 1:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    state["start"] = (x, y)
    state["gd_path"] = None
    state["mc_path"] = None
    stop_anim()
    start_dot.set_data([x], [y])
    gd_line.set_data([], [])
    gd_dot.set_data([], [])
    mc_line.set_data([], [])
    mc_dot.set_data([], [])
    gd_loss_line.set_data([], [])
    mc_loss_line.set_data([], [])
    info_text.set_text(f"Start: ({x:.2f}, {y:.2f})\nPress Run GD or Run MC-GD.")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", on_click)

# ── 타이틀 ───────────────────────────────────────────────────────────────────
fig.suptitle("Gradient Descent  vs  Monte Carlo Gradient Descent",
             color=TXT, fontsize=12, fontweight="bold", y=0.99)

plt.show()
