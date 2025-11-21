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
from SPEECHTOTEXTTENSORFLOW.constants import *
from SPEECHTOTEXTTENSORFLOW.models.model import Transformer
from SPEECHTOTEXTTENSORFLOW.models.model_utils import CustomSchedule, DisplayOutputs

class ModelTrainer():

    def __init__(self, data_preprocessing_artifacats: DataPreprocessingArtifacts, model_trainer_config = ModelTrainerConfig) -> None:

        try:
            self.data_preprocessing_artifacats = data_preprocessing_artifacats
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise STTException(e, sys)

    def vectorizer(self) -> VectorizeChar:
        try:
            logging.info("vectorising the data")
            self.vectorizer = VectorizeChar(MAX_TARGET_LENGTH)
            return self.vectorizer
        except Exception as e:
            raise STTException(e, sys)

    def get_data(self):
        train_data = self.data_preprocessing_artifacats.train_data_path
        test_data = self.data_preprocessing_artifacats.test_data_path
        try:
            with open(train_data) as f:
                self.dt_train = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]

            with open(test_data) as f:
                self.dt_test = [{k: v for k, v in row.items()}
                    for row in csv.DictReader(f, skipinitialspace=True)]

        except Exception as e:
            raise STTException(e, sys)
    
    def get_tensors(self):
        try:

            self.ds = CreateTensors(data=self.dt_train, vectorizer=self.vectorizer).create_tf_dataset(bs=16)
            self.val_ds = CreateTensors(data=self.dt_test, vectorizer=self.vectorizer).create_tf_dataset(bs=4)
        except Exception as e:
            raise STTException(e, sys)

    def fit(self):
        try:
            logging.info('fit the model')
            batch = next(iter(self.val_ds))

            # The vocabulary to convert predicted indices into characters
            idx_to_char = self.vectorizer.get_vocabulary()
            display_cb = DisplayOutputs(
                batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
            )
            self.model = Transformer(
                num_hid=200,
                num_head=2,
                num_feed_forward=400,
                target_maxlen=MAX_TARGET_LENGTH,
                num_layers_enc=4,
                num_layers_dec=1,
                num_classes=34,
            )
            loss_fn = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=0.1,
            )

            learning_rate = CustomSchedule(
                init_lr=0.00001,
                lr_after_warmup=0.001,
                final_lr=0.00001,
                warmup_epochs=15,
                decay_epochs=40,
                steps_per_epoch=len(self.ds),
            )
            optimizer = keras.optimizers.Adam(learning_rate)
            self.model.compile(optimizer=optimizer, loss=loss_fn)

            self.model.fit(self.ds, validation_data=self.val_ds, callbacks=[display_cb], epochs=EPOCHS)       

        except Exception as e:
            raise STTException(e, sys)
    

    def initiate_model_trainer(self) -> None:
        try:
            self.vectorizer()
            self.get_data()
            self.get_tensors()
            self.fit()

            model_loss = self.model.val_loss.numpy()
            
            os.makedirs(self.model_trainer_config.model_dir_path, exist_ok=True)
            weights_path = os.path.join(self.model_trainer_config.model_dir_path, SAVED_MODEL_DIR)
            os.makedirs(weights_path, exist_ok=True)
            self.model.save_weights(weights_path)

            model_trianer_artifact = ModelTrainerArtifacts(
                model_path=weights_path,
                model_loss=model_loss
            )
            return model_trianer_artifact

        except Exception as e:
            raise STTException(e, sys)
            