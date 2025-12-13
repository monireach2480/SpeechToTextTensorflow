import os, sys
import tensorflow as tf

from SPEECHTOTEXTTENSORFLOW.models.model import Transformer
from SPEECHTOTEXTTENSORFLOW.utils import path_to_audio
from SPEECHTOTEXTTENSORFLOW.models.data_utils import VectorizeChar
from SPEECHTOTEXTTENSORFLOW.constants import MAX_TARGET_LENGTH
from SPEECHTOTEXTTENSORFLOW.logger import logging
from SPEECHTOTEXTTENSORFLOW.exceptions import STTException


class Prediction:
    def __init__(self, audio_path, model_path):
        try:
            self.vectorizer = VectorizeChar(MAX_TARGET_LENGTH)
            self.audio_path = audio_path
            self.model_path = model_path
        except Exception as e:
            raise STTException(e, sys)
    
    def prediction(self):
        try:
            idx_to_char = self.vectorizer.get_vocabulary()

            logging.info("vocabulary created")

            model = Transformer(
                num_hid=200,
                num_head=2,
                num_feed_forward=400,
                target_maxlen=MAX_TARGET_LENGTH,
                num_layers_enc=4,
                num_layers_dec=1,
                num_classes=34,
            )

            logging.info("model isntance created")

            model.load_weights(self.model_path)
            logging.info("model weights loaded")

            preds = model.generate(tf.expand_dims(path_to_audio(path=self.audio_path), axis=0), target_start_token_idx=2)

            preds = preds.numpy()

            prediction = ""
            for idx in preds[0]:
                prediction += idx_to_char[idx]
                if idx_to_char[idx] == '>':
                    break
            
            logging.info("Prediction completed")
            
            return str(prediction)
        except Exception as e:
            raise STTException(e, sys)