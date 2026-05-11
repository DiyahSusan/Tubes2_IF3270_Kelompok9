import json

from .layer import EmbeddingLayer, DenseLayer, LSTMLayer
from .model import LSTMFromScratch


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as file:
        word_to_idx = json.load(file)
    idx_to_word = {str(idx): word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def build_lstm_from_keras(
    keras_model,
    vocab_path,
    max_len=30,
    num_lstm_layers=1,
):
    word_to_idx, idx_to_word = load_vocab(vocab_path)
    embedding_weights = keras_model.get_layer("embedding").get_weights()[0]

    feature_projection_weights = keras_model.get_layer(
        "feature_projection"
    ).get_weights()

    output_weights = keras_model.get_layer(
        "output_dense"
    ).get_weights()

    embedding_layer = EmbeddingLayer(embedding_weights)
    feature_projection_layer = DenseLayer(
        feature_projection_weights,
        activation=None,
    )

    lstm_layers = []
    for i in range(num_lstm_layers):
        layer_name = "lstm" if i == 0 else f"lstm_{i}"
        keras_lstm_layer = keras_model.get_layer(layer_name)

        lstm_layer = LSTMLayer(
            units=keras_lstm_layer.units,
            return_sequences=True,
            weights=keras_lstm_layer.get_weights(),
        )
        lstm_layers.append(lstm_layer)

    output_layer = DenseLayer(
        output_weights,
        activation="softmax",
    )

    return LSTMFromScratch(
        embedding_layer=embedding_layer,
        feature_projection_layer=feature_projection_layer,
        lstm_layers=lstm_layers,
        output_layer=output_layer,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        max_len=max_len,
    )