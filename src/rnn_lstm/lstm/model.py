import numpy as np


class LSTMFromScratch:
    def __init__(
        self,
        embedding_layer,
        feature_projection_layer,
        lstm_layers,
        output_layer,
        word_to_idx,
        idx_to_word,
        max_len=30,
    ):
        self.embedding_layer = embedding_layer
        self.feature_projection_layer = feature_projection_layer
        self.lstm_layers = lstm_layers
        self.output_layer = output_layer

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.max_len = max_len

        self.start_id = word_to_idx["<start>"]
        self.end_id = word_to_idx["<end>"]

    def forward(self, cnn_feature, input_tokens):
        cnn_feature = cnn_feature.astype(np.float32)

        # x(-1): projected CNN feature
        image_embedding = self.feature_projection_layer.forward(cnn_feature)
        image_embedding = image_embedding.reshape(1, -1)

        # x(0), x(1), ...: word embeddings
        word_embeddings = self.embedding_layer.forward(input_tokens)

        # [CNN_feature, emb(<start>), emb(word_1), ...]
        sequence = np.concatenate(
            [image_embedding, word_embeddings],
            axis=0,
        )

        output = sequence
        for lstm_layer in self.lstm_layers:
            output = lstm_layer.forward(output)

        probabilities = self.output_layer.forward(output)
        return probabilities

    def greedy_decode(self, cnn_feature):
        token_ids = [self.start_id]
        caption_words = []

        for _ in range(self.max_len):
            input_tokens = np.array(token_ids, dtype=np.int64)
            probabilities = self.forward(cnn_feature, input_tokens)
            next_id = int(np.argmax(probabilities[-1]))

            if next_id == self.end_id:
                break

            token_ids.append(next_id)
            word = self.idx_to_word[str(next_id)]
            caption_words.append(word)
        return " ".join(caption_words)