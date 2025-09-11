# src/delayer.py
from __future__ import annotations
import numpy as np
from typing import Iterable, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

ArrayLike = Union[np.ndarray]

class Delayer(TransformerMixin, BaseEstimator):
    """
    Scikit-learn Transformer to add time-lagged copies of features.

    Parameters
    ----------
    delays : Iterable[int]
        List of non-negative integer lags (in samples/TRs). 0 keeps the original features.
        Example: [0, 1, 2, 3, 4].
    drop_initial : bool, default=True
        If True, the first max(delays) rows (which are undefined for positive lags)
        are dropped so that all output rows are valid. If False, those rows are kept
        and filled with `fill_value`.
    fill_value : float, default=0.0
        Used only when drop_initial=False. Value to fill for undefined leading rows.
    dtype : Optional[np.dtype], default=None
        If provided, cast the output to this dtype.

    Notes
    -----
    - Output shape:
        If X has shape (T, F) and `delays` has length L,
        then transform(X) has shape:
          • (T - max_delay, F * L) if drop_initial=True
          • (T,               F * L) if drop_initial=False
    - Feature ordering:
        Concatenates blocks per delay in ascending order. For delays=[0,1,2],
        columns are [X(t), X(t-1), X(t-2)] flattened across features.
    """

    def __init__(
        self,
        delays: Iterable[int],
        drop_initial: bool = True,
        fill_value: float = 0.0,
        dtype: Optional[np.dtype] = None,
    ):
        self.delays = list(delays)
        self.drop_initial = drop_initial
        self.fill_value = fill_value
        self.dtype = dtype

    # sklearn 1.5 uses the estimator method `_validate_data`
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        X = self._validate_data(
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype=None,
            reset=True,
        )
        self.n_features_in_ = X.shape[1]
        if len(self.delays) == 0:
            raise ValueError("`delays` must contain at least one non-negative integer.")
        if any(d < 0 for d in self.delays):
            raise ValueError("All `delays` must be non-negative integers.")
        self._sorted_delays_: List[int] = sorted(set(int(d) for d in self.delays))
        self._max_delay_: int = int(max(self._sorted_delays_))
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=None)
        if X.shape[1] != getattr(self, "n_features_in_", X.shape[1]):
            raise ValueError(
                f"Number of features in transform ({X.shape[1]}) != fit ({self.n_features_in_})."
            )

        T, F = X.shape
        max_d = self._max_delay_

        # Build lagged blocks
        blocks: List[np.ndarray] = []
        for d in self._sorted_delays_:
            if d == 0:
                block = X
            else:
                # shift down by d: row t contains X[t-d]
                pad = np.empty((d, F), dtype=X.dtype)
                pad[:] = np.nan  # placeholders for undefined entries
                block = np.vstack([pad, X[:-d, :]])
            blocks.append(block)

        Xlags = np.concatenate(blocks, axis=1)  # (T, F * L)

        if self.drop_initial:
            # drop first max_delay rows (undefined for positive lags)
            Xout = Xlags[max_d:, :]
        else:
            # fill NaNs in the first max_delay rows
            if np.isnan(Xlags[:max_d, :]).any():
                Xlags[:max_d, :][np.isnan(Xlags[:max_d, :])] = self.fill_value
            Xout = Xlags

        if self.dtype is not None:
            Xout = Xout.astype(self.dtype, copy=False)

        return Xout

    # Helpful for sklearn pipelines that rely on feature names
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        F = getattr(self, "n_features_in_", None)
        if F is None:
            raise AttributeError("Call fit before get_feature_names_out.")
        if input_features is None:
            input_features = [f"x{i}" for i in range(F)]
        names = []
        for d in self._sorted_delays_:
            names.extend([f"{name}_lag{d}" for name in input_features])
        return np.asarray(names, dtype=object)

    # Useful when aligning targets
    def trim_target(self, y: ArrayLike) -> np.ndarray:
        """
        Trim y to match transform(X) when drop_initial=True.
        If drop_initial=False, returns y unchanged.

        Parameters
        ----------
        y : array-like of shape (T,) or (T, k)

        Returns
        -------
        y_out : ndarray
            y[max_delay:] if drop_initial, else y.
        """
        y = np.asarray(y)
        return y[self._max_delay_ :] if self.drop_initial else y
