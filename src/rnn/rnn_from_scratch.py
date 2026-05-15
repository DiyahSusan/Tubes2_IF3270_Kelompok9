from __future__ import annotations
import json
import numpy as np

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def apply_activation(x: np.ndarray, activation: str | None) -> np.ndarray:
    if activation is None or activation == "linear":
        return x
    if activation == "tanh":
        return tanh(x)
    if activation == "relu":
        return relu(x)
    if activation == "softmax":
        return softmax(x)
    raise ValueError(f"Aktivasi belum didukung: {activation}")


class EmbeddingScratch:
    def __init__(self, weights: np.ndarray):
        self.W = weights.astype(np.float32)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = token_ids.astype(np.int64)
        return self.W[token_ids]


class DenseScratch:
    def __init__(self, W: np.ndarray, b: np.ndarray, activation: str | None = None):
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.matmul(x, self.W) + self.b
        return apply_activation(y, self.activation)


class SimpleRNNCellScratch:
    def __init__(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray,
                 activation: str = "tanh"):

        self.Wx = kernel.astype(np.float32)
        self.Wh = recurrent_kernel.astype(np.float32)
        self.b = bias.astype(np.float32)
        self.activation = activation
        self.units = self.b.shape[0]

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        h = x_t @ self.Wx + h_prev @ self.Wh + self.b
        return apply_activation(h, self.activation)


class SimpleRNNLayerScratch:
    def __init__(self, kernel: np.ndarray, recurrent_kernel: np.ndarray, bias: np.ndarray,
                 activation: str = "tanh", return_sequences: bool = True):
        self.cell = SimpleRNNCellScratch(kernel, recurrent_kernel, bias, activation)
        self.return_sequences = return_sequences
        self.units = self.cell.units

    def forward(self, x: np.ndarray, h0: np.ndarray | None = None):
        batch_size, seq_len, _ = x.shape
        if h0 is None:
            h_t = np.zeros((batch_size, self.units), dtype=np.float32)
        else:
            h_t = h0.astype(np.float32)

        outputs = []
        for t in range(seq_len):
            h_t = self.cell.step(x[:, t, :], h_t)
            outputs.append(h_t)

        output_seq = np.stack(outputs, axis=1)  
        if self.return_sequences:
            return output_seq, h_t
        return h_t, h_t

class RNNCaptionDecoderScratch:
    def __init__(self, feature_projection: DenseScratch, embedding: EmbeddingScratch,
                 rnn_layers: list[SimpleRNNLayerScratch], output_dense: DenseScratch,
                 idx_to_word: dict[int, str] | None = None,
                 word_to_idx: dict[str, int] | None = None):
        self.feature_projection = feature_projection
        self.embedding = embedding
        self.rnn_layers = rnn_layers
        self.output_dense = output_dense
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx

    def _make_preinject_sequence(self, image_features: np.ndarray, input_token_ids: np.ndarray) -> np.ndarray:
        feature_embed = self.feature_projection.forward(image_features)
        feature_embed = feature_embed[:, None, :] 

        word_embed = self.embedding.forward(input_token_ids) 
        x = np.concatenate([feature_embed, word_embed], axis=1)
        return x

    def forward_teacher_forcing(self, image_features: np.ndarray, input_token_ids: np.ndarray) -> np.ndarray:
        x = self._make_preinject_sequence(image_features, input_token_ids)

        # pass ke recurrent stack
        for rnn in self.rnn_layers:
            x, _ = rnn.forward(x)

        logits = self.output_dense.forward(x)
        return logits

    def predict_next_distribution(self, image_features: np.ndarray, input_token_ids: np.ndarray) -> np.ndarray:
        probs = self.forward_teacher_forcing(image_features, input_token_ids)
        return probs[:, -1, :]  # (batch, vocab_size)

    def generate_greedy(self, image_feature: np.ndarray, start_id: int, end_id: int,
                        max_len: int = 30, pad_id: int | None = None) -> list[int]:
        if image_feature.ndim == 1:
            image_feature = image_feature[None, :]

        tokens = [start_id]
        generated = []
        for _ in range(max_len):
            inp = np.array(tokens, dtype=np.int64)[None, :]
            next_prob = self.predict_next_distribution(image_feature, inp)[0]

            if pad_id is not None:
                next_prob[pad_id] = -1.0
            next_id = int(np.argmax(next_prob))

            if next_id == end_id:
                break
            generated.append(next_id)
            tokens.append(next_id)

        return generated

    def generate_caption(self, image_feature: np.ndarray, start_id: int, end_id: int,
                         max_len: int = 30, pad_id: int | None = None) -> str:
        ids = self.generate_greedy(image_feature, start_id, end_id, max_len, pad_id)
        if self.idx_to_word is None:
            return " ".join(map(str, ids))
        return " ".join(self.idx_to_word.get(i, "<unk>") for i in ids)

def build_scratch_decoder_from_keras(keras_model, layer_names: dict,
                                     idx_to_word: dict[int, str] | None = None,
                                     word_to_idx: dict[str, int] | None = None) -> RNNCaptionDecoderScratch:
    
    Wp, bp = keras_model.get_layer(layer_names["feature_projection"]).get_weights()
    feature_projection = DenseScratch(Wp, bp, activation=None)

    We = keras_model.get_layer(layer_names["embedding"]).get_weights()[0]
    embedding = EmbeddingScratch(We)

    rnn_layers = []
    for name in layer_names["rnn_layers"]:
        kernel, recurrent_kernel, bias = keras_model.get_layer(name).get_weights()
        rnn_layers.append(
            SimpleRNNLayerScratch(
                kernel, recurrent_kernel, bias,
                activation="tanh",
                return_sequences=True
            )
        )

    Wo, bo = keras_model.get_layer(layer_names["output_dense"]).get_weights()
    output_dense = DenseScratch(Wo, bo, activation="softmax")

    return RNNCaptionDecoderScratch(
        feature_projection=feature_projection,
        embedding=embedding,
        rnn_layers=rnn_layers,
        output_dense=output_dense,
        idx_to_word=idx_to_word,
        word_to_idx=word_to_idx,
    )