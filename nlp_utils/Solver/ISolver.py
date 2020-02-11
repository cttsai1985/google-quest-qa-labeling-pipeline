from typing import Dict, Optional


class ISolver:
    @property
    def fine_tuned_model_weights_file_path_(self) -> str:
        raise NotImplementedError()

    def run(self, data: Dict, fit_params: Optional[Dict] = None, inference_only: bool = False, **kwargs):
        raise NotImplementedError()

    def analyze(self):
        raise NotImplementedError()

    @property
    def test_prediction_(self):
        raise NotImplementedError()

    @property
    def valid_trues_(self):
        raise NotImplementedError()

    @property
    def valid_prediction_(self):
        raise NotImplementedError()

    @property
    def valid_score_(self) -> float:
        raise NotImplementedError()
