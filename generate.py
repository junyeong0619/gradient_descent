import numpy as np


def generate_dataset(n_samples: int, w_true: float, b_true: float,
                     noise_std: float, seed: int = None) -> tuple:
    """
    몬테카를로 샘플링으로 선형 데이터셋 생성.

    y_i = w_true * x_i + b_true + epsilon_i
    epsilon_i ~ N(0, noise_std^2)

    Parameters
    ----------
    n_samples  : 데이터 포인트 수
    w_true     : 실제 기울기
    b_true     : 실제 절편
    noise_std  : 가우시안 노이즈 표준편차 (sigma)
    seed       : 재현성을 위한 난수 시드

    Returns
    -------
    x : shape (n_samples,)
    y : shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(low=0.0, high=10.0, size=n_samples)

    # 몬테카를로 샘플링: N(0, sigma^2) 노이즈 주입
    epsilon = rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    y = w_true * x + b_true + epsilon

    return x, y


def monte_carlo_datasets(n_trials: int, n_samples: int,
                         w_true: float, b_true: float,
                         noise_std: float) -> list:
    """
    서로 다른 랜덤 시드로 n_trials개의 독립 데이터셋 생성.
    MC 반복 실험을 통해 파라미터 수렴 분포를 분석하기 위해 사용.

    Returns
    -------
    datasets : list of (x, y) tuples, length = n_trials
    """
    datasets = []
    for trial in range(n_trials):
        x, y = generate_dataset(n_samples, w_true, b_true, noise_std, seed=trial)
        datasets.append((x, y))
    return datasets
