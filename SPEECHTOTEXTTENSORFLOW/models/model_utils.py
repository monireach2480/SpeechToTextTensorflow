import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from STT.exceptions import STTException
from STT.logger import logging


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        try:
            super().__init__()
            self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)
        except Exception as e:
            raise STTException(e, sys)

    def call(self, x):
        try:
            maxlen = tf.shape(x)[-1]
            x = self.emb(x)
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            return x + positions
        except Exception as e:
            raise STTException(e, sys)

class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        try:
            super().__init__()
            self.conv1 = tf.keras.layers.Conv1D(
                num_hid, 11, strides=2, padding="same", activation="relu"
            )
            self.conv2 = tf.keras.layers.Conv1D(
                num_hid, 11, strides=2, padding="same", activation="relu"
            )
            self.conv3 = tf.keras.layers.Conv1D(
                num_hid, 11, strides=2, padding="same", activation="relu"
            )
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)
        except Exception as e:
            raise STTException(e, sys)

    def call(self, x):
        try:
            x = self.conv1(x)
            x = self.conv2(x)
            return self.conv3(x)
        except Exception as e:
            raise STTException(e, sys)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        try:
            super().__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = keras.Sequential(
                [
                    layers.Dense(feed_forward_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)
        except Exception as e:
            raise STTException(e, sys)

    def call(self, inputs, training):
        try:
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
        except Exception as e:
            raise(e, sys)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        try:
            super().__init__()
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
            self.self_att = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
            self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.self_dropout = layers.Dropout(0.5)
            self.enc_dropout = layers.Dropout(0.1)
            self.ffn_dropout = layers.Dropout(0.1)
            self.ffn = keras.Sequential(
                [
                    layers.Dense(feed_forward_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )
        except Exception as e:
            raise STTException(e, sys)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        try:
            i = tf.range(n_dest)[:, None]
            j = tf.range(n_src)
            m = i >= j - n_src + n_dest
            mask = tf.cast(m, dtype)
            mask = tf.reshape(mask, [1, n_dest, n_src])
            mult = tf.concat(
                [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
            )
            return tf.tile(mask, mult)
        except Exception as e:
            raise STTException(e, sys)

    def call(self, enc_out, target):
        try:
            input_shape = tf.shape(target)
            batch_size = input_shape[0]
            seq_len = input_shape[1]
            causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
            target_att = self.self_att(target, target, attention_mask=causal_mask)
            target_norm = self.layernorm1(target + self.self_dropout(target_att))
            enc_out = self.enc_att(target_norm, enc_out)
            enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
            ffn_out = self.ffn(enc_out_norm)
            ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
            return ffn_out_norm
        except Exception as e:
            raise STTException(e, sys)

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch

        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        try:
            self.batch = batch
            self.target_start_token_idx = target_start_token_idx
            self.target_end_token_idx = target_end_token_idx
            self.idx_to_char = idx_to_token
        except Exception as e:
            raise STTException(e, sys)

    def on_epoch_end(self, epoch, logs=None):
        try:
            if epoch % 10 != 0:
                return
            source = self.batch["source"]
            target = self.batch["target"].numpy()
            bs = tf.shape(source)[0]
            preds = self.model.generate(source, self.target_start_token_idx)
            preds = preds.numpy()
            for i in range(bs):
                target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
                prediction = ""
                for idx in preds[i, :]:
                    prediction += self.idx_to_char[idx]
                    if idx == self.target_end_token_idx:
                        break
                print(f"target:     {target_text.replace('-','')}")
                print(f"prediction: {prediction}\n")
        except Exception as e:
            raise STTException(e, sys)

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        try:
            super().__init__()
            self.init_lr = init_lr
            self.lr_after_warmup = lr_after_warmup
            self.final_lr = final_lr
            self.warmup_epochs = warmup_epochs
            self.decay_epochs = decay_epochs
            self.steps_per_epoch = steps_per_epoch
        except Exception as e:
            raise STTException(e, sys)

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        try:
            warmup_lr = (
                self.init_lr
                + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
            )
            decay_lr = tf.math.maximum(
                self.final_lr,
                self.lr_after_warmup
                - (epoch - self.warmup_epochs)
                * (self.lr_after_warmup - self.final_lr)
                / (self.decay_epochs),
            )
            return tf.math.minimum(warmup_lr, decay_lr)
        except Exception as e:
            raise STTException(e, sys)

    def __call__(self, step):
        try:
            epoch = step // self.steps_per_epoch
            return self.calculate_lr(epoch)
        except Exception as e:
            raise STTException(e, sys)