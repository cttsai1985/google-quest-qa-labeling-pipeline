from typing import Optional, Callable, Dict, List, Tuple
from functools import partial
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
# from tensorflow.keras.callbacks import ModelCheckpoint

from ..Augmentation import AugmentationMaster
from .BaseTransformerTFSolver import BaseTransformerTFSolver
from ..Callback import CustomMetricEarlyStoppingCallback


def learning_rate_scheduler(epoch: int, lr: float, max_lr: float = 5e-4, factor: float = .5):
    lr_scheduled = tf.math.minimum(max_lr, lr * tf.math.exp(factor * epoch))
    if epoch > 0:
        print(f"\nNext epoch {epoch + 1}: previous learning rate: {lr:.6f} - scheduled to {lr_scheduled: .6f}")
    return lr_scheduled


def custom_loss(y_true, y_pred):
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)(y_true, y_pred)
    return bce_loss - cosine_loss


class BaselineTransformerTFSolver(BaseTransformerTFSolver):
    def __init__(
            self, fine_tuned_dir: str, score_func: Callable, encode_func: Callable, configs: Dict,
            cv_splitter: Optional = None, ):
        super().__init__(
            fine_tuned_dir=fine_tuned_dir, score_func=score_func, batch_encode_func=encode_func, configs=configs)

        self.cv_splitter = cv_splitter
        self.loss_direction: str = configs.get("loss_direction", 'auto')
        self.eval_metric = 'val_loss'

    def _model_fit(self, data, train_idx, model, validation_data, fit_params):
        train_x = data.get('train_x', None)
        train_y = data.get('train_y', None)

        # TODO: generator and data augmentation
        train_outputs = train_y.iloc[train_idx].values
        train_inputs = self._batch_encode(train_x.iloc[train_idx])

        print("\nTraining classification head block only")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_crossentropy', 'mse', 'mae'])
        fit_params_init = fit_params.copy()
        fit_params_init['epochs'] = 2
        model.fit(train_inputs, train_outputs, **fit_params_init)

        print("\nFine tune the whole model w/ early stopping")
        model.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
        model.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_crossentropy', 'mse', 'mae'])

        # callbacks
        warmup_lr_scheduler = partial(learning_rate_scheduler, max_lr=1e-4, factor=.25)
        lr_schedule = LearningRateScheduler(warmup_lr_scheduler)
        reduce_lr = ReduceLROnPlateau(
            monitor=self.eval_metric, factor=0.5, patience=2, min_lr=1e-6, model=self.loss_direction)
        early_stopping = CustomMetricEarlyStoppingCallback(
            data=validation_data, training_data=(train_inputs, train_outputs), score_func=self.score_func, patience=2,
            verbose=1, mode="auto", restore_best_weights=True)
        tensorboard = TensorBoard(self.fine_tuned_dir_path)
        callbacks = [early_stopping, reduce_lr, lr_schedule, tensorboard]  # model_checkpoint: got NotImplementedError
        model.fit(train_inputs, train_outputs, validation_data=validation_data, callbacks=callbacks, **fit_params)
        return self

    def _run_model_fine_tune(self, data: Dict, fit_params: Optional[Dict] = None, **kwargs):
        train_x = data.get('train_x', None)
        train_y = data.get('train_y', None)
        train_groups = data.get('train_groups', None)

        # TODO: adding HPO in the future
        # FIXME: for now it only runs single model not cv models
        for fold, (train_idx, valid_idx) in enumerate(self.cv_splitter.split(
                X=train_y, y=train_groups['category'].cat.codes, groups=None), start=1):
            self.tokenizer, model = self._pipeline_factory(
                load_model_from_fine_tuned=False, output_size=len(self.target_columns))

            valid_outputs = train_y.iloc[valid_idx].values
            valid_inputs = self._batch_encode(train_x.iloc[valid_idx])

            # training
            self._model_fit(
                data, train_idx, model, validation_data=(valid_inputs, valid_outputs), fit_params=fit_params)
            model.save_weights(self.fine_tuned_model_weights_file_path_)

            preds = model.predict(valid_inputs)
            self.valid_score = self.score_func(valid_outputs, preds)
            self.preds_valid = pd.DataFrame(preds, index=train_x.iloc[valid_idx].index, columns=self.target_columns)
            self.trues_valid = train_y.iloc[valid_idx]
            print(f'best validation metric score: {self.valid_score:.3f}')
            break  # FIXME: for now it only runs single model not cv models
        #
        return self


from tensorflow.keras.utils import Sequence


class TokenizedSequence(Sequence):
    def __init__(
            self, batch_encode_func: Callable, tokenizer, configs_question: Dict, configs_answer: Dict,
            x_set: pd.DataFrame, y_set: pd.DataFrame, is_distilled: bool = False, func_x_list: Optional[Tuple] = None,
            func_y_list: Optional[Tuple] = None, batch_size: int = 8, random_seed: int = 42, ):

        self.rng = np.random.RandomState(random_seed)

        self.x: pd.DataFrame = x_set
        self.y: pd.DataFrame = y_set
        self.batch_size: int = batch_size

        #
        self.tokenizer = tokenizer
        self._batch_encode_func: Callable = batch_encode_func
        self.configs_question: Dict = configs_question
        self.configs_answer: Dict = configs_answer
        self.is_distilled: bool = is_distilled

        #
        self.transformers = AugmentationMaster(func_x_list, func_y_list)

        self._gen_sequence()

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx: int):
        pos_start = idx * self.batch_size

        batch_y = self.y.iloc[pos_start:pos_start + self.batch_size].copy().apply(
            lambda y: self.transformers.transform(y=y)).values

        batch_x = self.x.iloc[pos_start:pos_start + self.batch_size].copy()
        q_col = self.configs_question["column"]
        q_col_pair = self.configs_question.get("column_pair", None)
        a_col = self.configs_answer["column"]
        a_col_pair = self.configs_answer.get("column_pair", None)

        for col in [q_col_pair, a_col_pair]:
            if col is None:
                continue

            batch_x[col] = batch_x[col].str.split().apply(
                lambda x: " ".join(self.transformers.transform(x)))

        batch_x = self._batch_encode_func(
            batch_x, self.tokenizer, column_question=q_col, column_question_pair=q_col_pair,
            tokenize_config_question=self.configs_question["tokenize"], column_answer=a_col,
            column_answer_pair=a_col_pair, tokenize_config_answer=self.configs_answer["tokenize"],
            is_distilled=self.is_distilled)

        return batch_x, batch_y

    def _gen_sequence(self):
        sequence = self.y.index.tolist()
        self.rng.shuffle(sequence)
        self.x = self.x.reindex(index=sequence)
        self.y = self.y.reindex(index=sequence)
        return self

    def on_epoch_end(self):
        self._gen_sequence()
        return self


class AugmentedTransformerTFSolver(BaselineTransformerTFSolver):
    def __init__(
            self, fine_tuned_dir: str, score_func: Callable, encode_func: Callable, configs: Dict,
            cv_splitter: Optional = None, ):
        super().__init__(
            fine_tuned_dir=fine_tuned_dir, score_func=score_func, encode_func=encode_func, configs=configs,
            cv_splitter=cv_splitter)

        # augmentation
        self.func_x_list = configs.get("func_x_list", list())
        self.func_y_list = configs.get("func_y_list", list())

    def _model_fit(self, data: Dict, train_idx: List[int], model, validation_data, fit_params):
        train_x = data.get('train_x', None)
        train_y = data.get('train_y', None)

        # TODO: generator and data augmentation
        train_outputs = train_y.iloc[train_idx].values
        train_inputs = self._batch_encode(train_x.iloc[train_idx])

        print("\nTraining classification head block only")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_crossentropy', 'mse', 'mae'])
        fit_params_init = fit_params.copy()
        fit_params_init['epochs'] = 2
        model.fit(train_inputs, train_outputs, **fit_params_init)

        print("\nFine tune the whole model")
        model.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
        model.compile(
            loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_crossentropy', 'mse', 'mae'])

        # FIXME: data generator for tf training somehow broken, here is a hot fix
        augmented_iter = 10
        batch_size = 512  # len(train_idx)
        generator_train = TokenizedSequence(
            batch_encode_func=self._batch_encode_func, tokenizer=self.tokenizer, configs_question=self.configs_question,
            configs_answer=self.configs_answer, is_distilled=self.is_distilled, x_set=train_x.iloc[train_idx],
            y_set=train_y.iloc[train_idx], func_x_list=self.func_x_list, func_y_list=self.func_y_list,
            batch_size=batch_size, random_seed=42)

        fit_params_warmup = fit_params.copy()
        fit_params_warmup['epochs'] = 1
        warmup_lr_scheduler = partial(learning_rate_scheduler, max_lr=2e-5, factor=.1)
        lr_schedule = LearningRateScheduler(warmup_lr_scheduler)
        for i in range(augmented_iter):
            print(f"iteration {i + 1:03d}: warm up with augmentation")
            for j in range(len(generator_train)):
                t_x, t_y = generator_train[j]
                model.fit(t_x, t_y, callbacks=[lr_schedule], **fit_params_warmup)

            generator_train.on_epoch_end()
        # FIXME: data generator for tf training somehow broken, here is a hot fix # END

        print("\nFine tune the whole model w/ early stopping")
        model.trainable = True
        # callbacks
        warmup_lr_scheduler = partial(learning_rate_scheduler, max_lr=1e-4, factor=.25)
        lr_schedule = LearningRateScheduler(warmup_lr_scheduler)
        reduce_lr = ReduceLROnPlateau(
            monitor=self.eval_metric, factor=0.5, patience=2, min_lr=1e-6, model=self.loss_direction)
        early_stopping = CustomMetricEarlyStoppingCallback(
            data=validation_data, training_data=(train_inputs, train_outputs), score_func=self.score_func, patience=2,
            verbose=1, mode="auto", restore_best_weights=True)
        tensorboard = TensorBoard(self.fine_tuned_dir_path)
        callbacks = [early_stopping, reduce_lr, lr_schedule, tensorboard]  # model_checkpoint: got NotImplementedError
        model.fit(train_inputs, train_outputs, validation_data=validation_data, callbacks=callbacks, **fit_params)
        return self
