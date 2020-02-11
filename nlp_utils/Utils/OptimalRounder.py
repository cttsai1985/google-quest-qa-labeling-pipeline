from typing import Callable, Dict, List, Tuple
from functools import partial
import numpy as np
import pandas as pd

from scipy.stats import spearmanr


def spearmanr_corr(y_true: np.array, y_pred: np.array):
    return spearmanr(y_true, y_pred).correlation


class IRounder:
    def fit(self, y_ref: pd.DataFrame, y_pred: pd.DataFrame):
        raise NotImplementedError()

    def fit_transform(self, y_ref: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def transform(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class _OptimalRounder(IRounder):
    def __init__(self, ref: pd.DataFrame, iter: int = 1000, seed: int = 42):
        self.ref: pd.DataFrame = ref
        self.coef_: Dict[str, List[float]] = dict()
        self.value_: Dict[str, List[float]] = dict()
        self.iter: int = iter
        self.rng = np.random.RandomState(seed)

    def _evaluate(self, coef: np.array, y_true: pd.Series, y_pred: pd.Series, mapped_values: List[float]) -> float:
        raise NotImplementedError

    def _fit_one_column(self, ref: pd.Series, y_true: pd.Series, y_pred: pd.Series) -> Tuple[List[float], List[float]]:
        initial_coef = np.linspace(0, 1, num=ref.nunique())
        mapped_value = sorted(ref.unique())
        loss_partial = partial(self._evaluate, y_true=y_true, y_pred=y_pred, mapped_value=mapped_value)

        score = loss_partial(initial_coef)
        best_score = score
        best_solution = initial_coef
        len_x = len(initial_coef)
        for i in range(self.iter):
            solution = sorted(self.rng.rand(len_x))
            score = loss_partial(solution)
            if score is not None and score < best_score:
                best_score = score
                best_solution = solution

        return best_solution, mapped_value

    def _transform_one_column(self, y_pred: pd.Series, coef: List[float], mapped_value: List[float]) -> List[float]:
        len_map = len(mapped_value) - 1
        return list(map(lambda ind: mapped_value[min(ind, len_map)], np.digitize(np.nan_to_num(y_pred), bins=coef)))

    def fit(self, y_ref: pd.DataFrame, y_pred: pd.DataFrame):
        self.fit_transform(y_true=y_ref, y_pred=y_pred)
        return self

    def fit_transform(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        for col in y_true.columns:
            print(f'fitting: {col}')
            self.coef_[col], self.value_[col] = self._fit_one_column(self.ref[col], y_true[col], y_pred[col])

        return self.transform(y_pred)

    def transform(self, y_pred: pd.DataFrame) -> pd.DataFrame:
        return y_pred.apply(
            lambda x: self._transform_one_column(x, coef=self.coef_[x.name], mapped_value=self.value_[x.name]))


class OptimalRounder(_OptimalRounder):
    def __init__(self, ref: pd.DataFrame, loss: Callable = spearmanr_corr, direction: str = 'auto'):
        super().__init__(ref=ref)
        self.loss: Callable = loss
        self.direction: str = direction  # support ['max', 'min', 'auto']
        if self.direction == 'auto':
            self.direction = 'max'

    def _evaluate(self, coef: np.array, y_true: pd.Series, y_pred: pd.Series, mapped_value: List[float]) -> float:
        y_pred_hat = self._transform_one_column(y_pred, coef=coef, mapped_value=mapped_value)
        score = self.loss(y_true.values, y_pred_hat)
        if self.direction == 'max':
            return score * -1.

        return score
