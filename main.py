import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from generate import generate_dataset, monte_carlo_datasets
from gradient_descent import gradient_descent

# ── 하이퍼파라미터 및 실험 설정 ──────────────────────────────────────────────
W_TRUE        = 3.0
B_TRUE        = 7.0
NOISE_STD     = 5.0
N_SAMPLES     = 200
LEARNING_RATE = 0.05
N_EPOCHS      = 2000
N_MC_TRIALS   = 100   # 몬테카를로 반복 실험 횟수


def run_single(seed: int = 42):
    """단일 실험: 데이터 생성 → 경사하강법 → 시각화"""

    # 1. 데이터 생성
    x, y = generate_dataset(N_SAMPLES, W_TRUE, B_TRUE, NOISE_STD, seed=seed)

    # 2. 무작위 초기 파라미터
    rng = np.random.default_rng(seed)
    w_init = rng.uniform(-2.0, 2.0)
    b_init = rng.uniform(-2.0, 2.0)

    # 3. 경사하강법 실행
    result = gradient_descent(x, y,
                              learning_rate=LEARNING_RATE,
                              n_epochs=N_EPOCHS,
                              w_init=w_init,
                              b_init=b_init)

    w_final = result["w_final"]
    b_final = result["b_final"]
    cost_history = result["cost_history"]
    w_history = result["w_history"]
    b_history = result["b_history"]

    print("=" * 50)
    print(f"  실제 파라미터   : w = {W_TRUE:.4f}, b = {B_TRUE:.4f}")
    print(f"  초기 파라미터   : w = {w_init:.4f}, b = {b_init:.4f}")
    print(f"  학습 결과       : w = {w_final:.4f}, b = {b_final:.4f}")
    print(f"  최종 비용 J     : {cost_history[-1]:.6f}")
    print(f"  w 오차          : {abs(w_final - W_TRUE):.6f}")
    print(f"  b 오차          : {abs(b_final - B_TRUE):.6f}")
    noise_floor = (NOISE_STD ** 2) / 2
    print(f"  이론적 비용 하한: {noise_floor:.6f}  (= sigma^2 / 2)")
    print(f"  수렴 여부       : {'수렴 완료' if abs(cost_history[-1] - noise_floor) < 1.0 else '미수렴'}")
    print("=" * 50)

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Monte Carlo + Gradient Descent Linear Regression", fontsize=14)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])   # 산점도 + 직선
    ax2 = fig.add_subplot(gs[0, 2])    # 비용 곡선
    ax3 = fig.add_subplot(gs[1, 0])    # w 수렴
    ax4 = fig.add_subplot(gs[1, 1])    # b 수렴
    ax5 = fig.add_subplot(gs[1, 2])    # w-b 경로

    x_line = np.linspace(x.min(), x.max(), 200)

    # (a) 산점도: 초기 직선 → 최적 직선 → 실제 직선
    ax1.scatter(x, y, s=15, alpha=0.5, color="steelblue", label="Data points")
    ax1.plot(x_line, w_init * x_line + b_init,
             "r--", linewidth=1.5, label=f"Initial  (w={w_init:.2f}, b={b_init:.2f})")
    ax1.plot(x_line, w_final * x_line + b_final,
             "orangered", linewidth=2.5, label=f"Learned  (w={w_final:.2f}, b={b_final:.2f})")
    ax1.plot(x_line, W_TRUE * x_line + B_TRUE,
             "g--", linewidth=1.5, label=f"True     (w={W_TRUE}, b={B_TRUE})")
    ax1.set_title("(a) Regression Line Convergence")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=8)

    # (b) 비용 함수 감소 곡선
    ax2.plot(cost_history, color="darkorange", linewidth=1.5)
    ax2.set_title("(b) Cost J(w, b) vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cost")
    ax2.set_yscale("log")

    # (c) w 수렴
    ax3.plot(w_history, color="royalblue", linewidth=1.5)
    ax3.axhline(W_TRUE, color="green", linestyle="--", linewidth=1, label=f"w_true = {W_TRUE}")
    ax3.set_title("(c) w Convergence")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("w")
    ax3.legend(fontsize=8)

    # (d) b 수렴
    ax4.plot(b_history, color="mediumseagreen", linewidth=1.5)
    ax4.axhline(B_TRUE, color="green", linestyle="--", linewidth=1, label=f"b_true = {B_TRUE}")
    ax4.set_title("(d) b Convergence")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("b")
    ax4.legend(fontsize=8)

    # (e) 파라미터 공간에서의 이동 경로
    ax5.plot(w_history, b_history, color="purple", linewidth=1, alpha=0.7)
    ax5.scatter(w_history[0],  b_history[0],  color="red",   s=60, zorder=5, label="Start")
    ax5.scatter(w_history[-1], b_history[-1], color="green", s=60, zorder=5, label="End")
    ax5.scatter(W_TRUE, B_TRUE, color="gold", marker="*", s=150, zorder=6, label="True")
    ax5.set_title("(e) Parameter Space Trajectory")
    ax5.set_xlabel("w")
    ax5.set_ylabel("b")
    ax5.legend(fontsize=8)

    plt.savefig("result_single.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: result_single.png")


def run_monte_carlo():
    """MC 반복 실험: N_MC_TRIALS개의 독립 실험으로 파라미터 수렴 분포 분석"""

    datasets = monte_carlo_datasets(N_MC_TRIALS, N_SAMPLES, W_TRUE, B_TRUE, NOISE_STD)

    w_results = []
    b_results = []

    for trial, (x, y) in enumerate(datasets):
        rng = np.random.default_rng(seed=trial + 1000)
        w_init = rng.uniform(-2.0, 2.0)
        b_init = rng.uniform(-2.0, 2.0)

        result = gradient_descent(x, y,
                                  learning_rate=LEARNING_RATE,
                                  n_epochs=N_EPOCHS,
                                  w_init=w_init,
                                  b_init=b_init)
        w_results.append(result["w_final"])
        b_results.append(result["b_final"])

    w_results = np.array(w_results)
    b_results = np.array(b_results)

    print(f"\n[MC {N_MC_TRIALS}회 반복 실험 결과]")
    print(f"  w: mean={w_results.mean():.4f}  std={w_results.std():.4f}  (true={W_TRUE})")
    print(f"  b: mean={b_results.mean():.4f}  std={b_results.std():.4f}  (true={B_TRUE})")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Monte Carlo Distribution of Learned Parameters ({N_MC_TRIALS} trials)", fontsize=13)

    for ax, values, true_val, label in zip(
        axes,
        [w_results, b_results],
        [W_TRUE, B_TRUE],
        ["w", "b"]
    ):
        ax.hist(values, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(true_val, color="red", linestyle="--", linewidth=2,
                   label=f"{label}_true = {true_val}")
        ax.axvline(values.mean(), color="orange", linestyle="-", linewidth=2,
                   label=f"mean = {values.mean():.3f}")
        ax.set_title(f"Distribution of learned {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("result_mc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: result_mc.png")


if __name__ == "__main__":
    print("\n[1/2] Single run experiment")
    run_single(seed=42)

    print("\n[2/2] Monte Carlo repeated experiment")
    run_monte_carlo()
