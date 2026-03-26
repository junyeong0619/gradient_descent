import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 하이퍼파라미터 (시각화에 최적화된 고정값) ──
LR = 0.1
N_EPOCHS = 150
UPDATE_EVERY = 3   # N 에포크마다 화면 갱신


# ── 데이터 정규화 ──
def normalize(arr):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    return (arr - mean) / std, mean, std


def denormalize(arr_norm, mean, std):
    return arr_norm * std + mean


# ── 제너레이터: 한 에포크마다 현재 상태를 yield ──
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


# ── Plotly 차트 생성 ──
def make_pred_chart(Y_true, Y_pred, epoch, output_idx=0):
    y_true = Y_true[:, output_idx]
    y_pred = Y_pred[:, output_idx]
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color="royalblue", opacity=0.7, size=7),
        name="데이터 포인트"
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(color="red", dash="dash", width=1.5),
        name="완벽한 예측 (y=x)"
    ))
    fig.update_layout(
        title=f"예측값 vs 실제값  (Epoch {epoch})",
        xaxis_title="실제값",
        yaxis_title="예측값",
        height=400,
        legend=dict(x=0.02, y=0.95),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_loss_chart(epochs, losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=losses,
        mode="lines",
        line=dict(color="orange", width=2),
        name="MSE Loss"
    ))
    fig.update_layout(
        title="Loss 곡선",
        xaxis_title="Epoch",
        yaxis_title="MSE",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ── Streamlit UI ──
st.set_page_config(page_title="GD Simulator", layout="wide")
st.title("Gradient Descent Simulator")
st.caption(f"학습률: {LR}  |  에포크: {N_EPOCHS}  |  데이터 자동 정규화 적용")

# 설정 행
col_in, col_out, col_file = st.columns([1, 1, 3])
with col_in:
    n_inputs = st.number_input("입력 수", min_value=1, max_value=50, value=1)
with col_out:
    n_outputs = st.number_input("출력 수", min_value=1, max_value=10, value=1)
with col_file:
    uploaded = st.file_uploader(
        "CSV 파일을 드래그 앤 드롭하거나 클릭해서 업로드",
        type=["csv"],
        label_visibility="visible"
    )

# 파일 업로드 후 처리
if uploaded is not None:
    has_header = st.checkbox("첫 번째 행이 헤더(컬럼명)입니다", value=False)
    df = pd.read_csv(uploaded, header=0 if has_header else None)

    st.write(f"**로드된 데이터:** {df.shape[0]}행 × {df.shape[1]}열")

    expected_cols = n_inputs + n_outputs
    if df.shape[1] != expected_cols:
        st.error(
            f"컬럼 수가 맞지 않습니다. "
            f"입력({n_inputs}) + 출력({n_outputs}) = {expected_cols}개 컬럼이 필요한데 "
            f"현재 {df.shape[1]}개입니다."
        )
        st.stop()

    with st.expander("데이터 미리보기", expanded=False):
        st.dataframe(df.head(10))

    X_raw = df.iloc[:, :n_inputs].values.astype(float)
    Y_raw = df.iloc[:, n_inputs:].values.astype(float)

    # 출력이 여러 개면 어느 출력을 차트에 표시할지 선택
    output_display_idx = 0
    if n_outputs > 1:
        output_display_idx = st.selectbox(
            "차트에 표시할 출력 컬럼",
            options=list(range(n_outputs)),
            format_func=lambda i: f"출력 {i + 1} (컬럼 {n_inputs + i + 1})"
        )

    if st.button("학습 시작", type="primary"):
        X, X_mean, X_std = normalize(X_raw)
        Y, Y_mean, Y_std = normalize(Y_raw)

        # 메트릭 + 차트 플레이스홀더
        m1, m2 = st.columns(2)
        ph_epoch = m1.empty()
        ph_loss = m2.empty()

        c1, c2 = st.columns(2)
        ph_pred = c1.empty()
        ph_loss_chart = c2.empty()

        loss_history = []
        epoch_history = []

        for epoch, W, b, loss, pred_norm in gd_generator(X, Y, LR, N_EPOCHS):
            loss_history.append(loss)
            epoch_history.append(epoch)

            if epoch % UPDATE_EVERY == 0 or epoch == N_EPOCHS:
                ph_epoch.metric("Epoch", f"{epoch} / {N_EPOCHS}")
                ph_loss.metric("MSE Loss", f"{loss:.6f}")

                pred_denorm = denormalize(pred_norm, Y_mean, Y_std)

                ph_pred.plotly_chart(
                    make_pred_chart(Y_raw, pred_denorm, epoch, output_display_idx),
                    use_container_width=True
                )
                ph_loss_chart.plotly_chart(
                    make_loss_chart(epoch_history, loss_history),
                    use_container_width=True
                )

        st.success(f"학습 완료!  최종 Loss: {loss_history[-1]:.6f}")

        # 최종 파라미터 표시 (역정규화 적용)
        st.subheader("학습된 파라미터")
        input_labels = [f"x{i + 1}" for i in range(n_inputs)]
        output_labels = [f"y{j + 1}" for j in range(n_outputs)]

        param_rows = []
        for j in range(n_outputs):
            row = {input_labels[i]: float(W[i, j] * Y_std[j] / X_std[i]) for i in range(n_inputs)}
            row["bias (b)"] = float(denormalize(b[j], Y_mean[j], Y_std[j]))
            param_rows.append(row)

        st.dataframe(
            pd.DataFrame(param_rows, index=output_labels),
            use_container_width=True
        )
