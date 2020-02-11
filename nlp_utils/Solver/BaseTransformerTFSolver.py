from typing import Callable, Dict, Optional

import pandas as pd

import tensorflow.keras.backend as K
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

from .BaseTransformerSovler import MixinTransformerSolver
from .TransformerModelFactory import create_model_from_pretrained


class BaseTransformerTFSolver(MixinTransformerSolver):
    def __init__(self, fine_tuned_dir: str, score_func: Callable, batch_encode_func: Callable, configs: Dict, ):
        super().__init__(
            score_func=score_func, fine_tuned_dir=fine_tuned_dir, pretrained_dir=configs["pretrained_model_dir"],
            model_weights_filename=configs["model_weights_filename"])

        self._batch_encode_func = batch_encode_func

        self.configs_question: Dict = configs["question"]
        self.configs_answer: Dict = configs["answer"]

        self.is_distilled: bool = configs.get("is_distilled", False)

        self.special_tokens_dict = configs.get("special_tokens_dict", dict())
        self.tokenizer: AutoTokenizer = None
        self.model = None

        self.max_seq_length_question = self.configs_question["tokenize"]["max_length"]
        self.max_seq_length_answer = self.configs_answer["tokenize"]["max_length"]

    def _pipeline_factory(self, load_model_from_fine_tuned: bool = False, output_size: int = None):
        # FIXME: AutoTokenizer, AutoConfig, AutoTFModel has unexpected issue while load from self.fine_tuned_dir_path
        load_from_dir_path: str = self.pretrained_dir_path

        tokenizer = AutoTokenizer.from_pretrained(load_from_dir_path)
        tokenizer.save_pretrained(self.fine_tuned_dir_path)
        num_added_toks = tokenizer.add_special_tokens(self.special_tokens_dict)
        if len(self.special_tokens_dict) > 0:
            print(f"adding special {num_added_toks} tokens: {self.special_tokens_dict}")

        if output_size is None:
            raise ValueError("need to specified output size for create model")

        # init a new model for this
        model_configs = AutoConfig.from_pretrained(load_from_dir_path)
        model_configs.output_hidden_states = False  # Set to True to obtain hidden states

        print(f"load pretrained weights for transformer from: {load_from_dir_path}")
        K.clear_session()
        model_block = TFAutoModel.from_pretrained(load_from_dir_path, config=model_configs)
        # model_block.resize_token_embeddings(len(tokenizer))  # FIXME: transformer not implemented in TF
        model_block.save_pretrained(self.fine_tuned_dir_path)
        model = create_model_from_pretrained(
            model_block, max_seq_length_question=self.max_seq_length_question,
            max_seq_length_answer=self.max_seq_length_answer, output_size=output_size, is_distilled=self.is_distilled)

        if load_model_from_fine_tuned:
            print(f"load fine-tuned wieghts from : {self.fine_tuned_model_weights_file_path_}")
            model.load_weights(self.fine_tuned_model_weights_file_path_)

        return tokenizer, model

    def _batch_encode(self, x: pd.DataFrame):
        inputs = self._batch_encode_func(
            x, self.tokenizer, column_question=self.configs_question["column"],
            column_question_pair=self.configs_question.get("column_pair", None),
            tokenize_config_question=self.configs_question["tokenize"], column_answer=self.configs_answer["column"],
            column_answer_pair=self.configs_answer.get("column_pair", None),
            tokenize_config_answer=self.configs_answer["tokenize"], is_distilled=self.is_distilled)
        return inputs

    def _run_inference(self, x: pd.DataFrame):
        if self.model is None or self.tokenizer is None:
            self.tokenizer, self.model = self._pipeline_factory(
                load_model_from_fine_tuned=True, output_size=len(self.target_columns))
        else:
            print("inference using current loaded tokenizer and model")
        return pd.DataFrame(self.model.predict(self._batch_encode(x)), index=x.index, columns=self.target_columns)

    def _run_model_fine_tune(self, data: Dict, fit_params: Optional[Dict] = None, **kwargs):
        raise NotImplementedError()
