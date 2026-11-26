import numpy as np
import numpy.typing as npt

__version__: str

def simd_level() -> str: ...
def scan(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], valid: npt.NDArray[np.bool_]
) -> tuple[bool, int, int, float, float]: ...
def rdp_keep(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],  # (k, n), C-contiguous
    valid: npt.NDArray[np.bool_],
    eps: float,
    vertical: bool,
    max_points: int,
    keep: npt.NDArray[np.bool_],
) -> float: ...
