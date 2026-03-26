import tkinter as tk
from tkinter import ttk, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── 시각화에 최적화된 고정 하이퍼파라미터 ──
LR = 0.1
N_EPOCHS = 150
UPDATE_EVERY = 3   # N 에포크마다 화면 갱신
DELAY_MS = 40      # 갱신 간격 (ms)
MAX_SCATTER = 3000  # scatter에 표시할 최대 포인트 수 (대용량 데이터 대응)


def normalize(arr):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    return (arr - mean) / std, mean, std


def gd_generator(X, Y, lr, n_epochs):
    n, n_in = X.shape
    n_out = Y.shape[1]
    W = np.zeros((n_in, n_out))
    b = np.zeros(n_out)

    for epoch in range(1, n_epochs + 1):
        pred = X @ W + b
        residuals = pred - Y
        loss = float(np.mean(residuals ** 2))
        dW = (X.T @ residuals) / n
        db = residuals.mean(axis=0)
        W -= lr * dW
        b -= lr * db
        yield epoch, W.copy(), b.copy(), loss, pred.copy()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Gradient Descent Simulator")
        self.root.geometry("1100x620")
        self.root.resizable(True, True)

        self.gen = None
        self.X_raw = None
        self.Y_raw = None
        self.Y_mean = None
        self.Y_std = None
        self.loss_history = []
        self.epoch_history = []
        self._after_id = None
        self._scatter_idx = None  # 대용량 데이터 샘플링 인덱스

        self._build_ui()

    # ── UI 구성 ──
    def _build_ui(self):
        # 왼쪽 설정 패널
        left = ttk.Frame(self.root, padding=14)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="입력 수 (X columns)").pack(anchor=tk.W)
        self.n_inputs = tk.IntVar(value=2)
        ttk.Spinbox(left, from_=1, to=50, textvariable=self.n_inputs, width=8).pack(anchor=tk.W, pady=(2, 12))

        ttk.Label(left, text="출력 수 (Y columns)").pack(anchor=tk.W)
        self.n_outputs = tk.IntVar(value=1)
        ttk.Spinbox(left, from_=1, to=10, textvariable=self.n_outputs, width=8).pack(anchor=tk.W, pady=(2, 16))

        # 드롭존
        self.drop_lbl = tk.Label(
            left, text="CSV 파일을\n여기에 드래그",
            width=20, height=6,
            relief=tk.DASHED, bg="#f5f5f5", fg="#888",
            font=("Arial", 10)
        )
        self.drop_lbl.pack(pady=4)

        if HAS_DND:
            self.drop_lbl.drop_target_register(DND_FILES)
            self.drop_lbl.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop_lbl.config(text="tkinterdnd2 없음\n파일 경로 직접 입력")
            path_row = ttk.Frame(left)
            path_row.pack(fill=tk.X, pady=4)
            self.path_var = tk.StringVar()
            ttk.Entry(path_row, textvariable=self.path_var, width=18).pack(side=tk.LEFT)
            ttk.Button(path_row, text="로드", command=lambda: self._load_csv(self.path_var.get())).pack(side=tk.LEFT, padx=2)

        self.file_lbl = ttk.Label(left, text="파일 없음", foreground="gray", wraplength=170)
        self.file_lbl.pack(pady=(4, 12))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        self.status_lbl = ttk.Label(left, text="", foreground="steelblue")
        self.status_lbl.pack(pady=2)
        self.epoch_lbl = ttk.Label(left, text="")
        self.epoch_lbl.pack()
        self.loss_lbl = ttk.Label(left, text="")
        self.loss_lbl.pack()

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.btn = ttk.Button(left, text="학습 시작", command=self._start, state=tk.DISABLED)
        self.btn.pack(pady=6, fill=tk.X)

        ttk.Label(left, text=f"LR={LR}  |  Epochs={N_EPOCHS}", foreground="gray").pack(pady=(8, 0))

        # 오른쪽 차트 패널
        right = ttk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=8)

        self.fig = Figure(figsize=(8.5, 5.2), dpi=95)
        self.ax_pred = self.fig.add_subplot(121)
        self.ax_loss = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_charts()

    def _init_charts(self):
        self.ax_pred.set_title("예측값 vs 실제값")
        self.ax_pred.set_xlabel("실제값")
        self.ax_pred.set_ylabel("예측값")
        self.ax_loss.set_title("Loss 곡선")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("MSE")
        self.canvas.draw()

    # ── 파일 드롭 & 로드 ──
    def _on_drop(self, event):
        path = event.data.strip().strip("{}")
        self._load_csv(path)

    def _load_csv(self, path):
        try:
            df = pd.read_csv(path, header=None)
            n_in = self.n_inputs.get()
            n_out = self.n_outputs.get()

            if df.shape[1] != n_in + n_out:
                messagebox.showerror(
                    "컬럼 수 불일치",
                    f"입력({n_in}) + 출력({n_out}) = {n_in + n_out}개 컬럼이 필요합니다.\n"
                    f"현재 파일: {df.shape[1]}개 컬럼"
                )
                return

            self.X_raw = df.iloc[:, :n_in].values.astype(float)
            self.Y_raw = df.iloc[:, n_in:].values.astype(float)

            # scatter 샘플링 인덱스 미리 계산
            n = len(self.X_raw)
            if n > MAX_SCATTER:
                self._scatter_idx = np.random.choice(n, MAX_SCATTER, replace=False)
            else:
                self._scatter_idx = np.arange(n)

            fname = path.replace("\\", "/").split("/")[-1]
            self.file_lbl.config(text=f"{fname}\n{df.shape[0]:,}행 × {df.shape[1]}열", foreground="green")
            self.btn.config(state=tk.NORMAL)
            self.status_lbl.config(text="파일 로드 완료")
        except Exception as e:
            messagebox.showerror("오류", str(e))

    # ── 학습 시작 ──
    def _start(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)

        self.loss_history = []
        self.epoch_history = []

        X, _, _ = normalize(self.X_raw)
        Y, self.Y_mean, self.Y_std = normalize(self.Y_raw)

        self.gen = gd_generator(X, Y, LR, N_EPOCHS)
        self.btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="학습 중...")
        self._step()

    # ── 스텝 루프 (after 기반 비동기) ──
    def _step(self):
        if self.gen is None:
            return

        try:
            for _ in range(UPDATE_EVERY):
                epoch, W, b, loss, pred_norm = next(self.gen)
                self.loss_history.append(loss)
                self.epoch_history.append(epoch)

            pred_denorm = pred_norm * self.Y_std + self.Y_mean
            self._update_charts(pred_denorm, epoch, loss)
            self.epoch_lbl.config(text=f"Epoch: {epoch} / {N_EPOCHS}")
            self.loss_lbl.config(text=f"Loss:  {loss:.6f}")

            self._after_id = self.root.after(DELAY_MS, self._step)

        except StopIteration:
            self.status_lbl.config(text="학습 완료!")
            self.btn.config(state=tk.NORMAL)
            self.gen = None

    # ── 차트 갱신 ──
    def _update_charts(self, pred_denorm, epoch, loss):
        idx = self._scatter_idx
        y_true = self.Y_raw[idx, 0]
        y_pred = pred_denorm[idx, 0]

        self.ax_pred.cla()
        self.ax_pred.scatter(y_true, y_pred, alpha=0.3, s=4, color="royalblue")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        self.ax_pred.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        self.ax_pred.set_title(f"예측 vs 실제  (Epoch {epoch})")
        self.ax_pred.set_xlabel("실제값")
        self.ax_pred.set_ylabel("예측값")

        self.ax_loss.cla()
        self.ax_loss.plot(self.epoch_history, self.loss_history, color="darkorange", linewidth=2)
        self.ax_loss.set_title("Loss 곡선")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("MSE")

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()


if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    App(root)
    root.mainloop()
