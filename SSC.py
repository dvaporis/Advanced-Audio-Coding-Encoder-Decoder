import numpy as np
from scipy.signal import lfilter

def SSC(frame_T: np.ndarray, next_frame_T : np.ndarray, prev_frame_type: str) -> str:
    # Sequence Segmentation Control for AAC.
    # Decide current frame type based on previous frame and next-frame ESH detection.
    if prev_frame_type == "LSS":
        return "ESH"
    elif prev_frame_type == "LPS":
        return "OLS"
    elif prev_frame_type == "OLS":
        # Per-channel ESH detection on the next frame
        next_is_esh = _is_esh_per_channel(next_frame_T)
        ch1 = _frame_type_from_prev(prev_frame_type, next_is_esh[0])
        ch2 = _frame_type_from_prev(prev_frame_type, next_is_esh[1])
        return _combine_frame_types(ch1, ch2)
    else :
        # Per-channel ESH detection on the next frame
        next_is_esh = _is_esh_per_channel(next_frame_T)
        ch1 = _frame_type_from_prev(prev_frame_type, next_is_esh[0])
        ch2 = _frame_type_from_prev(prev_frame_type, next_is_esh[1])
        return _combine_frame_types(ch1, ch2)


def _frame_type_from_prev(prev_frame_type: str, next_is_esh: bool) -> str:
    # Per-channel frame type from previous frame and next-frame ESH flag.
    if prev_frame_type == "LSS":
        return "ESH"
    if prev_frame_type == "LPS":
        return "OLS"
    if prev_frame_type == "OLS":
        return "LSS" if next_is_esh else "OLS"
    else:
        return "ESH" if next_is_esh else "LPS"


def _combine_frame_types(ch1: str, ch2: str) -> str:
    # Combine the two channel frame types using the provided decision table.
    table = {
        ("OLS", "OLS"): "OLS",
        ("OLS", "LSS"): "LSS",
        ("OLS", "ESH"): "ESH",
        ("OLS", "LPS"): "LPS",
        ("LSS", "LSS"): "LSS",
        ("LSS", "ESH"): "ESH",
        ("LSS", "LPS"): "ESH",
        ("ESH", "ESH"): "ESH",
        ("ESH", "LPS"): "ESH",
        ("LPS", "LPS"): "LPS",
    }
    if (ch1, ch2) in table:
        return table[(ch1, ch2)]
    elif (ch2, ch1) in table:
        return table[(ch2, ch1)]
    return "OLS"


def _is_esh_per_channel(next_frame_T: np.ndarray) -> np.ndarray:
    # Compute ESH detection separately for each channel.
    if next_frame_T.ndim != 2 or next_frame_T.shape[1] != 2:
        raise ValueError("next_frame_T must be a 2048x2 matrix")
    return np.array([
        _is_esh_channel(next_frame_T[:, 0]),
        _is_esh_channel(next_frame_T[:, 1]),
    ])


def _is_esh_channel(next_frame_channel: np.ndarray) -> bool:
    # Detect ESH on a single channel using the AAC transient detector rules.
    # Define the filter coefficients for H(z) = (0.7548 - 0.7548 z^-1) / (1 - 0.5095 z^-1)
    b = np.array([0.7548, -0.7548])
    a = np.array([1.0, -0.5095])

    # Filter with H(z)
    filtered = lfilter(b, a, next_frame_channel, axis=0)
    # Take middle 1152 samples and split into 8 segments
    mid = filtered[448:1600]

    energies = []
    for i in range(8):
        segment = mid[i * 128:(i + 1) * 128]
        sl2 = np.sum(segment ** 2)
        energies.append(sl2)

    # Attack value calculation and detection: sl^2 > 1e-3 and dsl^2 > 10 for any l=1..7
    for l in range(1, 8):
        sl2 = energies[l]
        if sl2 <= 1e-3:
            continue
        prev_mean = np.mean(energies[:l])
        if prev_mean > 0 and (sl2 / prev_mean) > 10:
            return True

    return False