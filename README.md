### About google-quest-qa-labeling-pipeline

This is a Kaggle competition aims to improving automated understanding of complex question answer content. The task is to predict 30 indices of each question and answer pair which labeled by human. It closed at early February 2020. More info: https://www.kaggle.com/c/google-quest-challenge

My proposed approach for this competition is to utilizing large transformers and transfer them to learn to predict this task. The main pipeline is build on transformers and keras/tensorflow. Transformers (https://github.com/huggingface/transformers) by hugging-face is a ready-to-use library for employing popular transformer model weights. It is a very straightforward way to apply language models for various tasks in a very short period of time. Keras/Tensorflow is the deep learning framework to build the pipeline.


### Requirement

The pipeline runs inside docker container. An Unix environment with modern GPU computing capacity for Docker (Tested on Ubuntu 18).


### Project structure

`nlp_util`: the main repository of ML code.

`script`: python codes to execute and training and predict bash script.

`notebook`: notebooks for visualization.

`docker`: Dockerfile and requirement.txt for building computing environment.

`configs`: store handful preset configs to training models.

`input`: store data set and models. This arrangement make it easier to port to the environment on kaggle server.


### Reproduce the results

##### step 1) build the using docker image for modeling

run `bash 00_init_build_docker.sh`. This will pull a tensorflow docker image with GPU computing capacity and install necessary packages defined in `requirements.txt`.


##### step 2) get the competition data

run `bash 01_download_data.sh` to get the competition data set `google-quest-challenge.zip`. It will decompress the downloaded zip file into `train.csv`, `test.csv` and `sample_submission.csv`.

Note this step needs kaggle api installed, please visit https://github.com/Kaggle/kaggle-api and have api key setup. 


##### step 3) data wrangling and understand data

run `bash 02_run_jupyter_notebook.sh` to launch jupyter notebook server and open `01_data_analysis.ipynb` for have a overview on training and test data set.


##### step 4) obtained pretrained weights from transfomers

run `bash 03_download_pretrain_model_weights.sh` to download model pretrained weights. Beware of storage since models are at ~500MB per.


##### step 5) training model and inference test set

run `bash 04_run_training_bert.sh` to run bert-base-uncased with preset.

run `04_run_training_distill_roberta.sh` to run distilled roberta with preset.

run `04_run_training_distill_roberta_aug.sh` to distilled roberta with preset and training with augmentation.

This run `tf_starter.py` and the pipeline is flexible to switch among difference embedding supported by transformer models provided by hugging face.


##### step 6) inference only

run `bash 05_run_inference_bert.sh` to run inference bert-base-uncased with preset and using the finetuned weights from step 5).


##### step 7) result analysis

use `02_result_analysis.ipynb` within `notebook` for result analysis.
or log from tensorboard by `06_use_tensorboard.sh` with `http://localhost:6006/` on browser.
