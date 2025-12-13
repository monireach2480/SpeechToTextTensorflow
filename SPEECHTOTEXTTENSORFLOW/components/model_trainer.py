# SPEECHTOTEXTTENSORFLOW/components/model_trainer.py

import os
import sys
import csv

import tensorflow as tf
from tensorflow import keras

from SPEECHTOTEXTTENSORFLOW.exceptions import STTException
from SPEECHTOTEXTTENSORFLOW.logger import logging
from SPEECHTOTEXTTENSORFLOW.entity.artifact_entity import DataPreprocessingArtifacts, ModelTrainerArtifacts
from SPEECHTOTEXTTENSORFLOW.entity.config_entity import ModelTrainerConfig
from SPEECHTOTEXTTENSORFLOW.entity.model_entity import CreateTensors
from SPEECHTOTEXTTENSORFLOW.models.data_utils import VectorizeChar
from SPEECHTOTEXTTENSORFLOW.models.model import Transformer
from SPEECHTOTEXTTENSORFLOW.models.model_utils import CustomSchedule, DisplayOutputs

# Safe constants
try:
    from SPEECHTOTEXTTENSORFLOW.constants import *
except:
    pass

BATCH_SIZE = globals().get('BATCH_SIZE', 32)
VAL_BATCH_SIZE = globals().get('VAL_BATCH_SIZE', 8)
EPOCHS = globals().get('EPOCHS', 30)
MAX_TARGET_LENGTH = globals().get('MAX_TARGET_LENGTH', 200)
NUM_CLASSES = globals().get('NUM_CLASSES', 34)
START_TOKEN_IDX = globals().get('START_TOKEN_IDX', 2)
END_TOKEN_IDX = globals().get('END_TOKEN_IDX', 3)
SAVED_MODEL_DIR = globals().get('SAVED_MODEL_DIR', "saved_model")


class ModelTrainer:
    def __init__(self, data_preprocessing_artifacts: DataPreprocessingArtifacts, model_trainer_config: ModelTrainerConfig):
        self.data_preprocessing_artifacts = data_preprocessing_artifacts
        self.model_trainer_config = model_trainer_config

    def vectorizer(self):
        logging.info("Creating vectorizer")
        self.vectorizer = VectorizeChar(MAX_TARGET_LENGTH)

    def get_data(self):
        train_path = self.data_preprocessing_artifacts.train_data_path
        test_path = self.data_preprocessing_artifacts.test_data_path

        try:
            def read_ljspeech_csv(path):
                data = []
                base_dir = os.path.dirname(path)
                with open(path, encoding="cp1252") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("|", 2)  # split only on first two |
                        if len(parts) < 3:
                            continue
                        wav_path = os.path.join(base_dir, parts[1])
                        text = parts[2].strip().lower()
                        data.append({"audio": wav_path, "text": text})
                return data

            self.dt_train = read_ljspeech_csv(train_path)
            self.dt_test = read_ljspeech_csv(test_path)

            logging.info(f"Loaded {len(self.dt_train)} train, {len(self.dt_test)} val samples")

        except Exception as e:
            raise STTException(e, sys)

    def get_tensors(self):
        try:
            self.ds = CreateTensors(data=self.dt_train, vectorizer=self.vectorizer)\
                      .create_tf_dataset(bs=BATCH_SIZE)
            self.val_ds = CreateTensors(data=self.dt_test, vectorizer=self.vectorizer)\
                        .create_tf_dataset(bs=VAL_BATCH_SIZE)
        except Exception as e:
            raise STTException(e, sys)

    def fit(self):
        try:
            logging.info("Starting training...")
            batch = next(iter(self.val_ds))
            idx_to_char = self.vectorizer.get_vocabulary()

            display_cb = DisplayOutputs(
                batch, idx_to_char,
                target_start_token_idx=START_TOKEN_IDX,
                target_end_token_idx=END_TOKEN_IDX
            )

            self.model = Transformer(
                num_hid=200,
                num_head=2,
                num_feed_forward=400,
                target_maxlen=MAX_TARGET_LENGTH,
                num_layers_enc=4,
                num_layers_dec=1,
                num_classes=NUM_CLASSES,
            )

            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
            lr = CustomSchedule(
                init_lr=0.00001,
                lr_after_warmup=0.001,
                final_lr=0.00001,
                warmup_epochs=10,
                decay_epochs=85,
                steps_per_epoch=len(self.ds),
            )
            optimizer = keras.optimizers.Adam(lr)

            self.model.compile(optimizer=optimizer, loss=loss_fn)

            history = self.model.fit(
                self.ds,
                validation_data=self.val_ds,
                callbacks=[display_cb],
                epochs=EPOCHS,
                verbose=1
            )

            self.final_val_loss = history.history['val_loss'][-1]

        except Exception as e:
            raise STTException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("STARTING TRAINING PIPELINE")
            self.vectorizer()
            self.get_data()
            self.get_tensors()
            self.fit()

            save_dir = os.path.join(self.model_trainer.model_dir_path, SAVED_MODEL_DIR)
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_weights(save_dir)

            artifact = ModelTrainerArtifacts(
                model_path=save_dir,
                model_loss=self.final_val_loss
            )

            logging.info(f"MODEL SAVED → {save_dir}")
            return artifact

        except Exception as e:
            raise STTException(e, sys)