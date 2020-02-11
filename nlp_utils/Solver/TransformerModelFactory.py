import tensorflow as tf


def _configure_pretrained_model_block(model, max_seq_length: int, is_distilled: bool = False):
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    embedding_index = 0
    if model.config.output_hidden_states:
        embedding_index = -1

    input_ids = tf.keras.layers.Input((max_seq_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input((max_seq_length,), dtype=tf.int32)
    if is_distilled:
        embedding = model(input_ids, attention_mask=attention_mask)[embedding_index]
        return (input_ids, attention_mask), embedding

    token_type_ids = tf.keras.layers.Input((max_seq_length,), dtype=tf.int32)
    embedding = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[embedding_index]
    return (input_ids, attention_mask, token_type_ids), embedding


def _res_net_block(
        input_data, filters: int = 64, kernel_size: int = 3, strides: int = 1, dilation_rate: int = 1,
        data_format='channels_first'):
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, activation='relu',
        padding='same', data_format=data_format)(input_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, activation=None,
        padding='same', data_format=data_format)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def wave_net_alike_arch(
        input_data, filters: int = 32, kernel_size: int = 3, strides: int = 1, data_format='channels_first'):
    pool_size: int = 8
    # block d1
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    d1 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    d1 = tf.keras.layers.MaxPooling1D(pool_size)(d1)

    # block d2
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    d2 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    d2 = tf.keras.layers.MaxPooling1D(pool_size)(d2)

    # block d4
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=4, data_format=data_format)
    d4 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=4, data_format=data_format)
    d4 = tf.keras.layers.MaxPooling1D(pool_size)(d4)

    # block d8
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=8, data_format=data_format)
    d8 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=8, data_format=data_format)
    d8 = tf.keras.layers.MaxPooling1D(pool_size)(d8)

    # hidden
    x = tf.keras.layers.Add()([d1, d2])
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    hidden_d1d2 = tf.keras.layers.MaxPooling1D(pool_size)(x)

    # hidden
    x = tf.keras.layers.Add()([d4, d8])
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    hidden_d4d8 = tf.keras.layers.MaxPooling1D(pool_size)(x)

    # hidden
    x = tf.keras.layers.Add()([hidden_d1d2, hidden_d4d8])
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


def shallow_wave_net_alike_arch(
        input_data, filters: int = 32, kernel_size: int = 3, strides: int = 1, data_format='channels_first'):
    pool_size: int = 8
    # block d1
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    d1 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    d1 = tf.keras.layers.MaxPooling1D(pool_size)(d1)

    # block d2
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=8, data_format=data_format)
    d2 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=4, data_format=data_format)
    d2 = tf.keras.layers.MaxPooling1D(pool_size)(d2)

    # hidden
    x = tf.keras.layers.Add()([d1, d2])
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


def shallow_res_net_alike_arch(
        input_data, filters: int = 32, kernel_size: int = 3, strides: int = 1, data_format='channels_first'):
    pool_size: int = 8

    # block d1
    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    d1 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    d1 = tf.keras.layers.MaxPooling1D(pool_size)(d1)

    # block d2
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    d2 = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    d2 = tf.keras.layers.MaxPooling1D(pool_size)(d2)

    # hidden
    x = tf.keras.layers.Add()([d1, d2])
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


def dense_block(
        input_data, filters: int = 32, kernel_size: int = 3, strides: int = 1, data_format='channels_first'):
    pool_size: int = 8

    x = _res_net_block(
        input_data, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=4, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=2, data_format=data_format)
    x = _res_net_block(
        x, filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=1, data_format=data_format)
    x = tf.keras.layers.MaxPooling1D(pool_size)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x


def create_model_from_pretrained(
        model, max_seq_length_question: int, max_seq_length_answer: int, output_size: int = 30,
        is_distilled: bool = False):
    model.trainable = False

    q_inputs, q_embed = _configure_pretrained_model_block(model, max_seq_length_question, is_distilled=is_distilled)
    a_inputs, a_embed = _configure_pretrained_model_block(model, max_seq_length_answer, is_distilled=is_distilled)

    if is_distilled:
        q_input_ids, q_attention_mask = q_inputs
        a_input_ids, a_attention_mask = a_inputs
        inputs = [q_input_ids, q_attention_mask, a_input_ids, a_attention_mask]
    else:
        q_input_ids, q_attention_mask, q_token_type_ids = q_inputs
        a_input_ids, a_attention_mask, a_token_type_ids = a_inputs
        inputs = [q_input_ids, q_attention_mask, q_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids]

    embed_process = dense_block  # make it a bit complex than GlobalAveragePooling1D
    q_embed = embed_process(q_embed)
    a_embed = embed_process(a_embed)

    subtracted = tf.keras.layers.Subtract()([q_embed, a_embed])
    x = tf.keras.layers.Concatenate()([q_embed, a_embed, subtracted])
    x = tf.keras.layers.Dense(x.shape[-1], activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # model.summary()  # debug purpose
    return model
