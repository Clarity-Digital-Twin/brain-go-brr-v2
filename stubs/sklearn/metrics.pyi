# Type stubs for sklearn.metrics (minimal subset for brain-brr)

from typing import overload
import numpy as np
from numpy.typing import ArrayLike

@overload
def roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    average: str = ...,
    sample_weight: ArrayLike | None = ...,
    max_fpr: float | None = ...,
    multi_class: str = ...,
    labels: ArrayLike | None = ...,
) -> float: ...

@overload
def roc_auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    average: None,
    sample_weight: ArrayLike | None = ...,
    max_fpr: float | None = ...,
    multi_class: str = ...,
    labels: ArrayLike | None = ...,
) -> np.ndarray: ...

def average_precision_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    average: str = ...,
    pos_label: int = ...,
    sample_weight: ArrayLike | None = ...,
) -> float: ...

def roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: int | str | None = ...,
    sample_weight: ArrayLike | None = ...,
    drop_intermediate: bool = ...,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
