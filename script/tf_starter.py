from typing import Callable, Dict, Optional, List, Tuple
import os
import sys
import argparse
import random

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit

import tensorflow as tf
import transformers

# Workaround to run on kaggle server
is_kaggle_server: bool = "kaggle" in os.getcwd().split("/")  # check if in kaggle server
EXTERNAL_UTILS_LIB = "../nlp_utils"
INPUT_DIR = "../input"
if not is_kaggle_server:
    sys.path.append(EXTERNAL_UTILS_LIB)
else:
    EXTERNAL_UTILS_LIB = "/kaggle/input/nlp_utils"
    sys.path.append(EXTERNAL_UTILS_LIB)

from nlp_utils import BaselineTransformerTFSolver
from nlp_utils import AugmentedTransformerTFSolver


def seed_everything(seed: int = 42):
    # Python/TF Seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)


def spearmanr_ignore_nan(trues: np.array, preds: np.array):
    return np.nanmean(
        [spearmanr(ta, pa).correlation for ta, pa in
         zip(np.transpose(trues), np.transpose(np.nan_to_num(preds)) + 1e-7)])


def batch_encode_sequence(
        df: pd.DataFrame, tokenizer, column_question: str, column_answer: str,
        column_question_pair: Optional[str] = None, tokenize_config_question: Optional[Dict] = None,
        column_answer_pair: Optional[str] = None, tokenize_config_answer: Optional[Dict] = None,
        is_distilled: bool = False):
    # FIXME: fix padding to max length not working
    encode_sequence = df[column_question]
    if column_question_pair is not None:
        encode_sequence = zip(df[column_question], df[column_question_pair])
    if tokenize_config_question is None:
        tokenize_config_question = dict()

    tokenized_question = tokenizer.batch_encode_plus(encode_sequence, **tokenize_config_question)
    q_input_ids = tokenized_question["input_ids"].numpy()
    q_attention_mask = tokenized_question["attention_mask"].numpy()
    q_token_type_ids = tokenized_question["token_type_ids"].numpy()

    # fix?
    max_length = tokenize_config_question["max_length"]
    if max_length != q_input_ids.shape[1]:
        appended_length = max_length - q_input_ids.shape[1]
        q_input_ids = np.pad(q_input_ids, ((0, 0), (0, appended_length)), constant_values=tokenizer.unk_token_id)

    if max_length != q_attention_mask.shape[1]:
        appended_length = max_length - q_attention_mask.shape[1]
        q_attention_mask = np.pad(q_attention_mask, ((0, 0), (0, appended_length)), constant_values=0)

    encode_sequence = df[column_answer]
    if column_answer_pair is not None:
        encode_sequence = zip(df[column_answer], df[column_answer_pair])
    if tokenize_config_answer is None:
        tokenize_config_answer = dict()

    tokenized_answer = tokenizer.batch_encode_plus(encode_sequence, **tokenize_config_answer)
    a_input_ids = tokenized_answer["input_ids"].numpy()
    a_attention_mask = tokenized_answer["attention_mask"].numpy()
    a_token_type_ids = tokenized_answer["token_type_ids"].numpy()

    # fix?
    max_length = tokenize_config_answer["max_length"]
    if max_length != a_input_ids.shape[1]:
        appended_length = max_length - a_input_ids.shape[1]
        a_input_ids = np.pad(a_input_ids, ((0, 0), (0, appended_length)), constant_values=tokenizer.unk_token_id)

    if max_length != a_attention_mask.shape[1]:
        appended_length = max_length - a_attention_mask.shape[1]
        a_attention_mask = np.pad(a_attention_mask, ((0, 0), (0, appended_length)), constant_values=0)

    # print(q_input_ids.shape, q_attention_mask.shape, a_input_ids.shape, a_attention_mask.shape)
    if is_distilled:
        return q_input_ids, q_attention_mask, a_input_ids, a_attention_mask

    return q_input_ids, q_attention_mask, q_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids


def process_read_dataframe(df: pd.DataFrame):
    bins = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    # group
    df["unique_id_question_body"] = df["question_body"].astype("category").cat.codes
    df["unique_id_question_body"] = df["category"].str.cat(df["unique_id_question_body"].astype("str"), sep="_")
    df["host_stem"] = df["host"].str.split(".").apply(lambda x: ".".join(x[-2:]))
    group_columns = ["category", "host_stem", "unique_id_question_body"]
    df[group_columns] = df[group_columns].astype("category")

    # corpus
    columns = ["question_title", "question_body", "answer"]
    for col in columns:
        df[f"count_{col}"] = df[col].str.split(" ").apply(lambda x: len(x)).astype(np.int32)

    df["count_question_title_body"] = (df["count_question_title"] + df["count_question_body"]).astype(np.int32)
    df["count_question_title_body_answer"] = (df["count_question_title_body"] + df["count_answer"]).astype(np.int32)
    stats_columns = [f"count_{col}" for col in columns] + [
        "count_question_title_body", "count_question_title_body_answer"]

    df_stats = df[stats_columns].describe(bins)
    df_stats_split = df.groupby("category")[stats_columns].apply(lambda x: x.describe(bins)).unstack(0).T

    # concat
    # df["question_title_body"] = df["question_title"].str.cat(others=df["question_body"], sep=" ")
    # columns = columns + ["question_title_body"]
    return df[columns], df[group_columns], df_stats, df_stats_split


def read_train_test(data_dir: str, index_name: str, inference_only: bool = False):
    # output_categories
    target_columns = [
        "question_asker_intent_understanding", "question_body_critical", "question_conversational",
        "question_expect_short_answer", "question_fact_seeking", "question_has_commonly_accepted_answer",
        "question_interestingness_others", "question_interestingness_self", "question_multi_intent",
        "question_not_really_a_question", "question_opinion_seeking", "question_type_choice", "question_type_compare",
        "question_type_consequence", "question_type_definition", "question_type_entity", "question_type_instructions",
        "question_type_procedure", "question_type_reason_explanation", "question_type_spelling",
        "question_well_written", "answer_helpful", "answer_level_of_information", "answer_plausible",
        "answer_relevance", "answer_satisfaction", "answer_type_instructions", "answer_type_procedure",
        "answer_type_reason_explanation", "answer_well_written"
    ]
    output_categories_question = list(
        filter(lambda x: x.startswith("question_"), target_columns))
    output_categories_answer = list(filter(lambda x: x.startswith("answer_"), target_columns))
    output_categories = output_categories_question + output_categories_answer

    df_test = pd.read_csv(os.path.join(data_dir, "test.csv")).set_index(index_name)
    test_x, test_groups, test_stats, test_stats_split = process_read_dataframe(df_test)
    print(f"test shape = {df_test.shape}\n{test_stats}\n")

    data = {
        "test_x": test_x, "test_groups": test_groups, "output_categories_question": output_categories_question,
        "output_categories_answer": output_categories_answer, "output_categories": output_categories
    }
    if inference_only:
        return data

    # training
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv")).set_index(index_name)

    # labels
    df_train[target_columns] = df_train[target_columns].astype(np.float32)
    train_y = df_train[output_categories]
    train_x, train_groups, train_stats, train_stats_split = process_read_dataframe(df_train)
    print(f"train shape = {df_train.shape}\n{train_stats}\n")
    print(f"Split by category: \n{train_stats_split}\nResorted\n{train_stats_split.swaplevel().sort_index()}\n")
    data.update({
        "train_x": train_x, "train_y": train_y, "train_groups": train_groups, "train_stats": train_stats,
        "train_stats_split": train_stats_split, "output_categories_question": output_categories_question,
        "output_categories_answer": output_categories_answer}
    )
    return data


def make_submission(preds: np.array, data_dir: str, index_name: str):
    df_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv")).set_index(index_name)
    df_sub[df_sub.columns] = preds[df_sub.columns]
    preds.index.name = index_name
    preds.to_csv("submission.csv", index=True)
    return preds


def _cv_splitter_factory(splitter_gen: str, params: Dict):
    if splitter_gen not in ["GroupKFold", "StratifiedKFold", "StratifiedShuffleSplit"]:
        err_msg = f'{splitter_gen} is not supported'
        raise ValueError(err_msg)

    return globals()[splitter_gen](**params)


def parse_command_line():
    default_data_dir: str = "../input/google-quest-challenge/"
    default_pretrained_w_root_dir = "../input/hugging_face_pretrained/"
    default_model_configs_path: str = "../configs/bert_configs.py"
    default_model_weight_filename: str = "tf_model_fine-tuned.h5"

    parser = argparse.ArgumentParser(
        description="Google Quest Q&A Bert Learner", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="folder for data")
    parser.add_argument(
        "--weights-root-dir", type=str, default=default_pretrained_w_root_dir,
        help="root folder for pretrained weights")
    parser.add_argument("--model-weights-filename", type=str, default=default_model_weight_filename,
        help="fine tuned filename for model weights")
    parser.add_argument("--configs", type=str, default=default_model_configs_path, help="path to model configs")
    parser.add_argument("--inference-only", action="store_true", default=False,  help="inference only")
    parser.add_argument(
        "--use-class-weights", action="store_true", default=False, help="weighted loss by class weight")
    parser.add_argument(
        "--training-augmentation", action="store_true", default=False, help="training with augmentation")
    args = parser.parse_args()
    return args


def initialize_configs(filename: str):
    if not os.path.exists(filename):
        raise ValueError("Spec file {spec_file} does not exist".format(spec_file=filename))

    module_name = filename.split(os.sep)[-1].replace('.', '')

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():

    args = parse_command_line()

    configs = initialize_configs(args.configs).configs
    configs["model_weights_filename"] = args.model_weights_filename

    fit_params = configs.get("fit_params", dict())
    pretrained_model_type: str = configs.get("model", "bert-base-uncased")
    configs["pretrained_model_dir"] = os.path.join(args.weights_root_dir, pretrained_model_type)
    if pretrained_model_type.find("distil") >= 0:
        configs["is_distilled"] = True

    INDEX_NAME = 'qa_id'
    ##
    q_max_length = configs["question"]["tokenize"]["max_length"]
    a_max_length = configs["answer"]["tokenize"]["max_length"]
    generated_working_dir = f"{pretrained_model_type}_q{q_max_length}_a{a_max_length}"

    data = read_train_test(data_dir=args.data_dir, index_name=INDEX_NAME, inference_only=args.inference_only)
    if args.use_class_weights:
        train_w = data['train_y'].sum()
        class_weight = (train_w.median() / train_w).apply(np.sqrt)
        fit_params["class_weight"] = {i: w for i, w in enumerate(class_weight)}
        generated_working_dir = f"{generated_working_dir}_weighted"
        print(f"class weights:\n{class_weight}")

    # cv setup
    splitter_configs = configs.get("cv_splitter")
    splitter = _cv_splitter_factory(
        splitter_gen=splitter_configs["splitter_gen"], params=splitter_configs["params"])

    solver_gen = BaselineTransformerTFSolver
    if args.training_augmentation:
        solver_gen = AugmentedTransformerTFSolver
        generated_working_dir = f"{generated_working_dir}_augmented"

    WORKING_DIR = os.path.join("../input", generated_working_dir)
    if args.training_augmentation:
        solver = solver_gen(
            fine_tuned_dir=WORKING_DIR, cv_splitter=splitter, score_func=spearmanr_ignore_nan,
            encode_func=batch_encode_sequence, configs=configs,)
        solver.run(data, fit_params=fit_params, inference_only=args.inference_only)
    else:
        solver = solver_gen(
            fine_tuned_dir=WORKING_DIR, cv_splitter=splitter, score_func=spearmanr_ignore_nan,
            encode_func=batch_encode_sequence, configs=configs)
        solver.run(data, fit_params=fit_params, inference_only=args.inference_only)

    test_result = solver.test_prediction_
    make_submission(test_result, data_dir=args.data_dir, index_name=INDEX_NAME)
    return


if "__main__" == __name__:
    print(f"tensorflow version: {tf.__version__}")
    print(f"transformers version: {transformers.__version__}")
    seed_everything()
    np.set_printoptions(suppress=True)
    main()
    # multi gpu setup;
    # tf.debugging.set_log_device_placement(True)
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #    main()
