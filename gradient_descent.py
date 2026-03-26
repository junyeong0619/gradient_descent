import numpy as np


def cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    MSE 비용 함수.

    J(w, b) = 1/(2N) * sum( (h(x_i) - y_i)^2 )
    """
    n = len(x)
    predictions = w * x + b
    residuals = predictions - y
    return (1 / (2 * n)) * np.dot(residuals, residuals)


def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple:
    """
    비용 함수의 편미분 계산.

    dJ/dw = 1/N * sum( (h(x_i) - y_i) * x_i )
    dJ/db = 1/N * sum( (h(x_i) - y_i) )
    """
    n = len(x)
    residuals = (w * x + b) - y
    dj_dw = (1 / n) * np.dot(residuals, x)
    dj_db = (1 / n) * np.sum(residuals)
    return dj_dw, dj_db


def gradient_descent(x: np.ndarray, y: np.ndarray,
                     learning_rate: float = 0.01,
                     n_epochs: int = 1000,
                     w_init: float = 0.0,
                     b_init: float = 0.0) -> dict:
    """
    경사하강법으로 선형 회귀 파라미터 최적화.

    w <- w - alpha * dJ/dw
    b <- b - alpha * dJ/db

    Parameters
    ----------
    x, y           : 학습 데이터
    learning_rate  : 학습률 (alpha)
    n_epochs       : 반복 횟수
    w_init, b_init : 초기 파라미터

    Returns
    -------
    dict with keys:
        w_final    : 최종 기울기
        b_final    : 최종 절편
        cost_history   : 에포크별 비용값 list
        w_history      : 에포크별 w 변화 list
        b_history      : 에포크별 b 변화 list
    """
    w = w_init
    b = b_init

    cost_history = []
    w_history = [w]
    b_history = [b]

    for _ in range(n_epochs):
        dj_dw, dj_db = gradients(x, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        cost_history.append(cost(x, y, w, b))
        w_history.append(w)
        b_history.append(b)

    return {
        "w_final": w,
        "b_final": b,
        "cost_history": np.array(cost_history),
        "w_history": np.array(w_history),
        "b_history": np.array(b_history),
    }
