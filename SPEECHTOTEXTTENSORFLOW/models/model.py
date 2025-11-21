import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from STT.models.model_utils import TokenEmbedding, SpeechFeatureEmbedding, TransformerEncoder, TransformerDecoder
from STT.logger import logging
from STT.exceptions import STTException



class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        try:
            super().__init__()
            self.loss_metric = keras.metrics.Mean(name="loss")
            self.num_layers_enc = num_layers_enc
            self.num_layers_dec = num_layers_dec
            self.target_maxlen = target_maxlen
            self.num_classes = num_classes

            self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
            self.dec_input = TokenEmbedding(
                num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
            )

            self.encoder = keras.Sequential(
                [self.enc_input]
                + [
                    TransformerEncoder(num_hid, num_head, num_feed_forward)
                    for _ in range(num_layers_enc)
                ]
            )

            for i in range(num_layers_dec):
                setattr(
                    self,
                    f"dec_layer_{i}",
                    TransformerDecoder(num_hid, num_head, num_feed_forward),
                )

            self.classifier = layers.Dense(num_classes)
        except Exception as e:
            raise STTException(e, sys)

    def decode(self, enc_out, target):
        try:
            y = self.dec_input(target)
            for i in range(self.num_layers_dec):
                y = getattr(self, f"dec_layer_{i}")(enc_out, y)
            return y
        except Exception as e:
            raise STTException(e, sys)

    def call(self, inputs):
        try:
            source = inputs[0]
            target = inputs[1]
            x = self.encoder(source)
            y = self.decode(x, target)
            return self.classifier(y)
        except Exception as e:
            raise STTException(e, sys)
    
    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        try:
            source = batch["source"]
            target = batch["target"]
            dec_input = target[:, :-1]
            dec_target = target[:, 1:]
            with tf.GradientTape() as tape:
                preds = self([source, dec_input])
                one_hot = tf.one_hot(dec_target, depth=self.num_classes)
                mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
                loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.loss_metric.update_state(loss)
            return {"loss": self.loss_metric.result()}
        except Exception as e:
            raise STTException(e, sys)
    
    @property
    def val_loss(self):
        return self.loss_metric.result()

    def test_step(self, batch):
        try:
            source = batch["source"]
            target = batch["target"]
            dec_input = target[:, :-1]
            dec_target = target[:, 1:]
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
            self.loss_metric.update_state(loss)
            return {"loss": self.loss_metric.result()}
        except Exception as e:
            raise STTException(e, sys)

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        try:
            bs = tf.shape(source)[0]
            enc = self.encoder(source)
            dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
            dec_logits = []
            for i in range(self.target_maxlen - 1):
                dec_out = self.decode(enc, dec_input)
                logits = self.classifier(dec_out)
                logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
                last_logit = tf.expand_dims(logits[:, -1], axis=-1)
                dec_logits.append(last_logit)
                dec_input = tf.concat([dec_input, last_logit], axis=-1)
            return dec_input
        except Exception as e:
            raise STTException(e, sys)