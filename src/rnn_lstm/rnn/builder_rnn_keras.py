import tensorflow as tf
from tensorflow.keras import layers, Model


def build_keras_rnn_decoder(
    feature_dim: int,
    vocab_size: int,
    max_input_len: int,
    embed_dim: int = 256,
    hidden_size: int = 256,
    num_rnn_layers: int = 1,
    learning_rate: float = 1e-3,
    pad_id: int = 0,
):
    image_feature_input = layers.Input(shape=(feature_dim,), name="image_feature_input")
    caption_token_input = layers.Input(shape=(max_input_len,), name="caption_token_input")

    feature_embed = layers.Dense(embed_dim, name="feature_projection")(image_feature_input)
    feature_embed = layers.Reshape((1, embed_dim), name="feature_as_timestep")(feature_embed)

    caption_embed = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,
        name="caption_embedding"
    )(caption_token_input)

    x = layers.Concatenate(axis=1, name="preinject_concat")(
        [feature_embed, caption_embed]
    )

    for i in range(num_rnn_layers):
        x = layers.SimpleRNN(
            hidden_size,
            activation="tanh",
            return_sequences=True,
            name=f"rnn_{i+1}"
        )(x)

    x = layers.Lambda(lambda t: t[:, 1:, :], name="remove_feature_timestep")(x)

    output = layers.Dense(
        vocab_size,
        activation="softmax",
        name="vocab_output"
    )(x)

    model = Model(
        inputs=[image_feature_input, caption_token_input],
        outputs=output,
        name=f"rnn_decoder_{num_rnn_layers}layer_{hidden_size}hidden"
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none"
    )

    def masked_sparse_categorical_crossentropy(y_true, y_pred):
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, pad_id), dtype=loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_sparse_categorical_crossentropy,
        metrics=[]
    )

    return model