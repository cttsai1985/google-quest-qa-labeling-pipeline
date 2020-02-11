import os
from typing import Callable, Optional, List, Dict
import pandas as pd
from scipy.stats import ks_2samp

from .ISolver import ISolver


def _ckeck_dir_path_exist(working_path: str):
    if os.path.exists(working_path) and not os.path.isfile(working_path):
        return True

    return False


def _mkdir_safe(working_path: str) -> bool:
    if not _ckeck_dir_path_exist(working_path):
        os.makedirs(working_path)
        return True

    return False


class MixinTransformerSolver(ISolver):
    def __init__(
            self, score_func: Callable, fine_tuned_dir: str, pretrained_dir: str, model_weights_filename: str,
            model_stats_filename: str = "model_stats.hdf5"):

        self.score_func: Callable = score_func

        self.model_weights_filename: str = model_weights_filename  # consider move to configs
        self.model_stats_filename: str = model_stats_filename

        _mkdir_safe(fine_tuned_dir)
        self.fine_tuned_dir_path: str = fine_tuned_dir
        print(f"working dir: {self.fine_tuned_dir_path}")
        self.pretrained_dir_path: str = pretrained_dir
        if not _ckeck_dir_path_exist(pretrained_dir):
            err_msg = f"pretrained dir path is not exists: {pretrained_dir}"
            raise ValueError(err_msg)

        self.target_columns: Optional[List[str]] = None

        # results
        self.preds_test: Optional[pd.DataFrame] = None
        self.preds_valid: Optional[pd.DataFrame] = None
        self.trues_valid: Optional[pd.DataFrame] = None
        self.valid_score: Optional[float] = None

        self.is_executed: bool = False

    def _analyze_score_dist(self, data: Dict):
        train_groups = data.get("train_groups", None)

        # validation-test overall diff
        ks_result = self.preds_test.apply(lambda x: ks_2samp(x.values, self.preds_valid[x.name].values), axis=0)
        ks_stats, p_value = list(zip(*(ks_result.tolist())))
        stats_diff = pd.concat([
            self.preds_test.mean().rename("test_mean"), self.preds_valid.mean().rename("valid_mean"),
            (self.preds_test.mean() - self.preds_valid.mean()).rename("mean_diff"),
            self.preds_test.mean().rename("test_std"), self.preds_valid.mean().rename("valid_std"),
            pd.Series(ks_stats, index=self.preds_test.columns).rename("ks_stats"),
            pd.Series(p_value, index=self.preds_test.columns).rename("p_value"), ], axis=1).sort_values("mean_diff")
        print(f"valid-test difference:\n{stats_diff.round(6)}\n")

        # validation performance
        valid_breakdown_metrics = pd.concat([
            (self.trues_valid - self.preds_valid).mean(axis=0).rename("bias"),
            (self.trues_valid - self.preds_valid).abs().mean(axis=0).rename("mae"),
            ((self.trues_valid - self.preds_valid) / self.trues_valid.mean()).abs().mean(axis=0).rename("mape"),
            self.trues_valid.apply(
                lambda x: x.corr(self.preds_valid[x.name], method="pearson"), axis=0).rename("pearson"),
            self.trues_valid.apply(
                lambda x: x.corr(self.preds_valid[x.name], method="spearman"), axis=0).rename("spearman"),
        ], axis=1).sort_values("spearman", ascending=True)
        print(f"validation breakdown metrics:\n{valid_breakdown_metrics.round(6)}\n")

        valid_overall_metrics = valid_breakdown_metrics.describe()
        print(f"validation overall metrics:\n{valid_overall_metrics.round(6)}\n")

        #
        output_categories_question = data.get("output_categories_question", None)
        output_categories_answer = data.get("output_categories_answer", None)
        if output_categories_question is not None and output_categories_answer is not None:
            y_valid_q = self.trues_valid[output_categories_question]
            p_valid_q = self.preds_valid[output_categories_question]
            valid_score_question = self.score_func(y_valid_q.values, p_valid_q.values)

            y_valid_a = self.trues_valid[output_categories_answer]
            p_valid_a = self.preds_valid[output_categories_answer]
            valid_score_answer = self.score_func(y_valid_a.values, p_valid_a.values)
            print(f"valid score on question: {valid_score_question:.3f}, answer: {valid_score_answer:.3f}\n")

        # analysis by groups
        groupby_obj = train_groups.reindex(index=self.trues_valid.index).groupby("category")
        group_valid_score = groupby_obj.apply(lambda x: self.score_func(
            self.trues_valid.reindex(index=x.index).values, self.preds_valid.reindex(
                index=x.index).values)).to_frame("score")
        print(f"group valid score: \n{group_valid_score}\n")
        group_valid_score.index = group_valid_score.index.tolist()  # categorical index casting to normal str

        stats_dict = {
            'test_preds': self.preds_test,
            'valid_preds': self.preds_valid,
            'valid_trues': self.trues_valid,
            "valid_test_stats_diff": stats_diff,
            "valid_breakdown_metrics": valid_breakdown_metrics,
            "valid_overall_metrics": valid_overall_metrics,
            "valid_group_score": group_valid_score,
        }

        return stats_dict

    @property
    def fine_tuned_model_weights_file_path_(self) -> str:
        return os.path.join(self.fine_tuned_dir_path, self.model_weights_filename)

    @property
    def fine_tuned_model_stats_file_path_(self) -> str:
        return os.path.join(self.fine_tuned_dir_path, self.model_stats_filename)

    def run(self, data: Dict, fit_params: Optional[Dict] = None, inference_only: bool = False, **kwargs):
        test_x = data.get("test_x", None)
        self.target_columns = data["output_categories"]

        if inference_only:
            self.is_executed = True
            self.preds_test = self._run_inference(test_x)
            print(f"test dist:\n{self.preds_test.describe().T}")
            return self

        self._run_model_fine_tune(data=data, fit_params=fit_params, **kwargs)
        self.preds_test = self._run_inference(test_x)
        print(f"test dist:\n{self.preds_test.describe().T}")
        self.is_executed = True

        results = self._analyze_score_dist(data)
        with pd.HDFStore(self.fine_tuned_model_stats_file_path_, mode="w") as store:
            for k, v in results.items():
                store.put(key=k, value=v)
                print(f"save stats: {k}, shape={v.shape}")

        return self

    def analyze(self):
        import pdb;
        pdb.set_trace()
        return self

    @property
    def test_prediction_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        return self.preds_test

    @property
    def valid_trues_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.trues_valid is None:
            print("no model validation in this run")

        return self.trues_valid

    @property
    def valid_prediction_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.preds_valid is None:
            print("no model validation in this run")

        return self.preds_valid

    @property
    def valid_score_(self) -> float:
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.valid_score is None:
            print("no model validation while run")

        return self.valid_score

    def _run_inference(self, test_x):
        raise NotImplementedError()

    def _run_model_fine_tune(self, data: Dict, fit_params: Dict, **kwargs):
        raise NotImplementedError()

