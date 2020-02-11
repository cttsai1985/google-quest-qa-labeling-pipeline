"""
fork THIS excellent downloader
https://www.kaggle.com/maroberti/transformers-model-downloader-pytorch-tf2-0
"""

from typing import Union
from pathlib import Path
import os

import transformers
from transformers import AutoConfig, AutoTokenizer, TFAutoModel


def transformers_model_dowloader(pretrained_model_name: str, working_dir: Union[str, Path], is_tf: bool = True) -> bool:
    model_class = None
    if is_tf:
        model_class = TFAutoModel

    NEW_DIR = working_dir / pretrained_model_name
    try:
        os.mkdir(NEW_DIR)
        print(f"Successfully created directory {NEW_DIR}")
    except OSError:
        print(f"Creation of directory {NEW_DIR} failed")

    print(f"Download model and tokenizer {pretrained_model_name}")
    transformer_model = model_class.from_pretrained(pretrained_model_name)
    transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    try:
        transformer_model.save_pretrained(NEW_DIR)
        transformer_tokenizer.save_pretrained(NEW_DIR)
        print(f"Save model and tokenizer {pretrained_model_name} in directory {NEW_DIR}")
    except:
        print(f"Save model and tokenizer {pretrained_model_name} in directory {NEW_DIR}: Failed")
        return False

    return True


def main():
    pretrained_model_name_list = [
        'bert-base-uncased',
        'bert-base-cased',
        'bert-large-cased',

        'distilbert-base-uncased',

        'albert-xxlarge-v2',
        'albert-xlarge-v2',
        'albert-large-v2',

        'roberta-base',
        'roberta-large',
        'roberta-large-mnli',
        'distilroberta-base',

        'distilbert-base-uncased',
    ]

    print(f'Transformers version {transformers.__version__}')  # Current version: 2.3.0
    WORKING_DIR = Path("../input/hugging_face_pretrained")
    try:
        os.mkdir(WORKING_DIR)
    except:
        pass

    for i, pretrained_model_name in enumerate(pretrained_model_name_list, start=1):
        print(i, '/', len(pretrained_model_name_list))
        transformers_model_dowloader(pretrained_model_name, WORKING_DIR, is_tf=True)

    return


if "__main__" == __name__:
    main()
