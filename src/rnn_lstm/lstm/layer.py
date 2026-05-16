import numpy as np


class EmbeddingLayer:
    def __init__(self, weights):
        self.weights = {
            "embedding": weights.astype(np.float32)
        }
        
    def forward(self, x):
        return self.weights["embedding"][x]


class DenseLayer:
    def __init__(self, weights, activation=None):
        self.weights = {
            "kernel": weights[0].astype(np.float32),
            "bias": weights[1].astype(np.float32),
        }
        self.activation = activation

    def _softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def forward(self, x):
        z = np.dot(x, self.weights["kernel"]) + self.weights["bias"]
        if self.activation == "softmax":
            return self._softmax(z)
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        
        return z


class LSTMLayer:
    def __init__(self, units, return_sequences=True, weights=None):
        self.units = units
        self.return_sequences = return_sequences
        self.weights = {}

        if weights is not None:
            self.weights["kernel"] = weights[0].astype(np.float32)
            self.weights["recurrent_kernel"] = weights[1].astype(np.float32)
            self.weights["bias"] = weights[2].astype(np.float32)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, x):
        single_input = False

        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
            single_input = True

        batch_size, seq_len, input_dim = x.shape

        h = np.zeros((batch_size, self.units), dtype=np.float32)
        c = np.zeros((batch_size, self.units), dtype=np.float32)

        outputs = []

        for t in range(seq_len):
            # x(t)Wx + h(t-1)Wh + b
            gates = (
                np.dot(x[:, t, :], self.weights["kernel"])
                + np.dot(h, self.weights["recurrent_kernel"])
                + self.weights["bias"]
            )

            i_gate = self._sigmoid(gates[:, :self.units])
            f_gate = self._sigmoid(gates[:, self.units:2 * self.units])
            c_tilde = self._tanh(gates[:, 2 * self.units:3 * self.units])
            o_gate = self._sigmoid(gates[:, 3 * self.units:])

            c = f_gate * c + i_gate * c_tilde
            h = o_gate * self._tanh(c)

            outputs.append(h.copy())
        outputs = np.stack(outputs, axis=1)

        if not self.return_sequences:
            outputs = h

        if single_input:
            outputs = outputs[0]

        return outputs