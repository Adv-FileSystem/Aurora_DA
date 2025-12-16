import numpy as np
from typing import Iterable, List, Tuple, Union, Optional, Callable


ArrayLike = Union[np.ndarray, Iterable[float]]


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x


def add_noise(x: ArrayLike, snr_db_range: Tuple[float, float] = (10.0, 30.0)) -> np.ndarray:
    """가우시안 노이즈 추가."""
    x = _to_1d_array(x)
    power = np.mean(x ** 2) + 1e-8
    snr_db = np.random.uniform(*snr_db_range)
    snr = 10.0 ** (snr_db / 10.0)
    noise_power = power / snr
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def scale_amplitude(x: ArrayLike, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """진폭 스케일링."""
    x = _to_1d_array(x)
    scale = np.random.uniform(*scale_range)
    return x * scale


def time_shift(x: ArrayLike, max_shift_ratio: float = 0.1) -> np.ndarray:
    """시간축 쉬프트 (roll)."""
    x = _to_1d_array(x)
    n = len(x)
    max_shift = int(n * max_shift_ratio)
    if max_shift < 1:
        return x.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift)


def time_stretch(x: ArrayLike, stretch_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """시간 스트레칭/압축 후 원래 길이로 리샘플링."""
    x = _to_1d_array(x)
    n = len(x)
    factor = np.random.uniform(*stretch_range)
    new_len = max(2, int(n * factor))
    t_orig = np.arange(n)
    t_new = np.linspace(0, n - 1, new_len)

    x_stretched = np.interp(t_new, t_orig, x)

    if new_len > n:
        return x_stretched[:n]
    elif new_len < n:
        pad_len = n - new_len
        return np.pad(x_stretched, (0, pad_len), mode="edge")
    else:
        return x_stretched


def permute_segments(x: ArrayLike, n_segments: int = 4) -> np.ndarray:
    """구간 퍼뮤테이션."""
    x = _to_1d_array(x)
    n = len(x)

    if n_segments <= 1 or n < n_segments:
        return x.copy()

    indices = np.linspace(0, n, n_segments + 1, dtype=int)
    segments = [x[indices[i]:indices[i + 1]] for i in range(n_segments)]

    perm = np.random.permutation(n_segments)
    permuted = [segments[i] for i in perm]
    return np.concatenate(permuted)


# ---------- 여기서부터 보완용 추가 증강 ----------

def random_crop_and_resize(
    x: ArrayLike,
    crop_ratio_range: Tuple[float, float] = (0.7, 1.0),
) -> np.ndarray:
    """연속 구간을 랜덤 크기로 잘라서 다시 원래 길이로 리샘플링."""
    x = _to_1d_array(x)
    n = len(x)
    min_r, max_r = crop_ratio_range
    min_len = max(2, int(n * min_r))
    max_len = max(min_len, int(n * max_r))
    seg_len = np.random.randint(min_len, max_len + 1)

    if seg_len >= n:
        return x.copy()

    start = np.random.randint(0, n - seg_len + 1)
    seg = x[start:start + seg_len]

    # 다시 원래 길이(n)로 리샘플링
    t_orig = np.arange(seg_len)
    t_new = np.linspace(0, seg_len - 1, n)
    return np.interp(t_new, t_orig, seg)


def baseline_wander(
    x: ArrayLike,
    freq_range: Tuple[float, float] = (0.05, 0.5),
    amp_factor_range: Tuple[float, float] = (0.05, 0.2),
) -> np.ndarray:
    """저주파 사인 파형을 더해 baseline drift 생성."""
    x = _to_1d_array(x)
    n = len(x)

    freq = np.random.uniform(*freq_range)
    amp_factor = np.random.uniform(*amp_factor_range)
    amp = amp_factor * np.std(x)

    t = np.linspace(0, 1, n)
    drift = amp * np.sin(2 * np.pi * freq * t)
    return x + drift


def random_dropout(
    x: ArrayLike,
    max_dropout_ratio: float = 0.2,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """연속 구간을 0 또는 지정 값으로 드롭아웃."""
    x = _to_1d_array(x)
    n = len(x)
    if max_dropout_ratio <= 0:
        return x.copy()

    max_len = int(n * max_dropout_ratio)
    if max_len < 1:
        return x.copy()

    drop_len = np.random.randint(1, max_len + 1)
    start = np.random.randint(0, n - drop_len + 1)

    x_aug = x.copy()
    if fill_value is None:
        fill_value = 0.0
    x_aug[start:start + drop_len] = fill_value
    return x_aug


class SignalAugmentor:
    """여러 증강 기법을 확률적으로 적용하는 클래스."""

    def __init__(
        self,
        # 기존 증강 확률
        p_noise: float = 0.7,
        p_scale: float = 0.5,
        p_shift: float = 0.5,
        p_stretch: float = 0.5,
        p_permute: float = 0.3,
        # 추가 증강 확률
        p_crop: float = 0.5,
        p_baseline: float = 0.4,
        p_dropout: float = 0.3,
        # 하이퍼파라미터
        noise_snr_db_range: Tuple[float, float] = (10.0, 30.0),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        max_shift_ratio: float = 0.1,
        stretch_range: Tuple[float, float] = (0.9, 1.1),
        n_segments: int = 4,
        crop_ratio_range: Tuple[float, float] = (0.7, 1.0),
        baseline_freq_range: Tuple[float, float] = (0.05, 0.5),
        baseline_amp_factor_range: Tuple[float, float] = (0.05, 0.2),
        max_dropout_ratio: float = 0.2,
        dropout_fill_value: Optional[float] = 0.0,
        # 재현성을 위한 seed (옵션)
        random_state: Optional[int] = None,
        # 커스텀 후처리 함수 (예: 클리핑, 정규화 등)
        postprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_shift = p_shift
        self.p_stretch = p_stretch
        self.p_permute = p_permute

        self.p_crop = p_crop
        self.p_baseline = p_baseline
        self.p_dropout = p_dropout

        self.noise_snr_db_range = noise_snr_db_range
        self.scale_range = scale_range
        self.max_shift_ratio = max_shift_ratio
        self.stretch_range = stretch_range
        self.n_segments = n_segments
        self.crop_ratio_range = crop_ratio_range
        self.baseline_freq_range = baseline_freq_range
        self.baseline_amp_factor_range = baseline_amp_factor_range
        self.max_dropout_ratio = max_dropout_ratio
        self.dropout_fill_value = dropout_fill_value

        self.postprocess = postprocess

        if random_state is not None:
            np.random.seed(random_state)

    def augment_once(self, x: ArrayLike) -> np.ndarray:
        """신호 x(1D)를 한 번 증강."""
        x_aug = _to_1d_array(x).copy()

        # 순서는 필요에 따라 조정 가능
        if np.random.rand() < self.p_noise:
            x_aug = add_noise(x_aug, self.noise_snr_db_range)

        if np.random.rand() < self.p_scale:
            x_aug = scale_amplitude(x_aug, self.scale_range)

        if np.random.rand() < self.p_shift:
            x_aug = time_shift(x_aug, self.max_shift_ratio)

        if np.random.rand() < self.p_stretch:
            x_aug = time_stretch(x_aug, self.stretch_range)

        if np.random.rand() < self.p_permute:
            x_aug = permute_segments(x_aug, self.n_segments)

        if np.random.rand() < self.p_crop:
            x_aug = random_crop_and_resize(x_aug, self.crop_ratio_range)

        if np.random.rand() < self.p_baseline:
            x_aug = baseline_wander(
                x_aug,
                freq_range=self.baseline_freq_range,
                amp_factor_range=self.baseline_amp_factor_range,
            )

        if np.random.rand() < self.p_dropout:
            x_aug = random_dropout(
                x_aug,
                max_dropout_ratio=self.max_dropout_ratio,
                fill_value=self.dropout_fill_value,
            )

        if self.postprocess is not None:
            x_aug = self.postprocess(x_aug)

        return x_aug

    def augment_batch(self, x: ArrayLike, n_aug: int = 5) -> List[np.ndarray]:
        """원본 신호 x 한 개에 대해 n_aug개의 증강 샘플 리스트 반환."""
        return [self.augment_once(x) for _ in range(n_aug)]
