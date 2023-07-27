import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import losses
from tensorflow.keras.optimizers import Adam
from generic_model import generic_model



# generic model contains generic methods for loading and storing a model
class RNN(generic_model):

    def __init__(self, config):

        super(RNN, self).__init__(config)

        # Store important parameters
        self.rnn_name = config['rnn']
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.embed_dim = config['embedding_dim']
        self.output_dim = config['vocab_size']

        # whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = layers.Embedding(self.input_dim, self.embed_dim)
        else:
            self.use_embedding = False

        # linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim * 2
        mid_features = config['output_mid_features']
        self.linear1_out = layers.Dense(mid_features)
        self.relu = layers.ReLU()
        self.linear2_out = layers.Dense(self.output_dim)

        # linear layer after missed characters
        self.miss_linear = layers.Dense(config['miss_linear_dim'])

        # declare RNN
        if self.rnn_name == 'LSTM':
            self.rnn = layers.LSTM(units=self.hidden_dim, return_sequences=True,
                                   dropout=config['dropout'],
                                   recurrent_dropout=config['dropout'])
        else:
            self.rnn = layers.GRU(units=self.hidden_dim, return_sequences=True,
                                  dropout=config['dropout'],
                                  recurrent_dropout=config['dropout'])

        # optimizer
        self.optimizer = Adam(learning_rate=config['lr'])

    def call(self, x, x_lens, miss_chars):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, input_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :param miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
        :return: tensor of shape (batch size, max sequence length, output dim)
        """
        if self.use_embedding:
            x = self.embedding(x)

        # Pack sequences
        x = tf.keras.layers.Masking(mask_value=0)(x)
        x = tf.keras.layers.RNN(self.rnn)(x)

        hidden = x[:, -1]  # Get the last output from the RNN

        # Project miss_chars onto a higher dimension
        miss_chars = self.miss_linear(miss_chars)

        # Concatenate RNN output and miss chars
        concatenated = tf.concat((hidden, miss_chars), axis=1)

        # Predict
        return self.linear2_out(self.relu(self.linear1_out(concatenated)))

    def calculate_loss(self, model_out, labels, input_lens, miss_chars):
        """
        :param model_out: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, vocab_size). 1 at index i indicates that ith character should be predicted
        :param: miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
                            passed here to check if model's output probability of missed_chars is decreasing
        """
        outputs = tf.nn.log_softmax(model_out, axis=1)

        # Calculate model output loss for miss characters
        miss_penalty = tf.reduce_sum(outputs * miss_chars, axis=(0, 1)) / tf.cast(tf.shape(outputs)[0], dtype=tf.float32)

        input_lens = tf.cast(input_lens, dtype=tf.float32)
        # Weights per example is inversely proportional to length of word
        # This is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = 1 / input_lens / tf.reduce_sum(1 / input_lens)
        weights = tf.expand_dims(weights_orig, axis=-1)

        # Actual loss
        loss_func = losses.BinaryCrossentropy(reduction=losses.Reduction.SUM, weight=weights)
        actual_penalty = loss_func(labels, model_out)

        return actual_penalty, miss_penalty
