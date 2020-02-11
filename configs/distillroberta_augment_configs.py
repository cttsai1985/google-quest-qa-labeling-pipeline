configs = {
    "func_x_list": [
        ("RandomTruncateHead",
         {"min_length": 128, "max_length": 256, "random_seed": 42, "threshold": 0.75}),
        ("RandomTruncateTail",
         {"min_length": 128, "max_length": 256, "random_seed": 42, "threshold": 0.75}),
        ("RandomDropWords",
         {"min_length": 64, "max_drop": 8, "drop_rate": .2, "random_seed": 42, "threshold": 0.5})
    ],

    "func_y_list": [
        ("LabelSoften", {"min_value": 0.1, "max_value": 0.9, "random_seed": 42, "threshold": 0.95})
    ],

    "model": "distilroberta-base",
    "cv_splitter": {
        "splitter_gen": "GroupKFold",
        "split_index": "unique_id_question_body",
        "params": {
            "n_splits": 5,
        },

    },

    "fit_params": {
        "batch_size": 8,
        "epochs": 10,
        "verbose": 1,
        # "callbacks": None,
        "shuffle": True,
        "steps_per_epoch": None,
        "validation_steps": None,
        "validation_freq": 1,
    },

    "special_tokens_dict": {},
    "question": {
        "column": "question_title",
        "column_pair": "question_body",
        "tokenize": {
            "add_special_tokens": True,
            "max_length": 384,  # 256,
            "stride": 0,
            "truncation_strategy": "longest_first",
            "return_tensors": "tf",
            "return_input_lengths": False,
            "return_attention_masks": True,
            "pad_to_max_length": True,
        },
    },

    "answer": {
        "column": "question_title",
        "column_pair": "answer",
        "tokenize": {
            "add_special_tokens": True,
            "max_length": 512,  # 384,
            "stride": 0,
            "truncation_strategy": "longest_first",
            "return_tensors": "tf",
            "return_input_lengths": False,
            "return_attention_masks": True,
            "pad_to_max_length": True,
        },
    },
}
