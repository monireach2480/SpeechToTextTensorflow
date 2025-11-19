import os
import sys
import csv
from glob import glob

from SPEECHTOTEXTTENSORFLOW.entity.config_entity import DataPreprocessingConfig
from SPEECHTOTEXTTENSORFLOW.entity.artifact_entity import DataPreprocessingArtifacts, DataIngestionArtifacts
from SPEECHTOTEXTTENSORFLOW.models.data_utils import VectorizeChar, get_data
from SPEECHTOTEXTTENSORFLOW.logger import logging
from SPEECHTOTEXTTENSORFLOW.exceptions import STTException
from SPEECHTOTEXTTENSORFLOW.constants import *


class DataPreprocessing():
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig, data_ingestion_artifact: DataIngestionArtifacts) -> None:
        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise STTException(e, sys)
    
    def get_id_to_text(self) -> tuple:
        try:
            logging.info("Entering the get_id_to_index method of DataPreprocessing")
            os.makedirs(self.data_preprocessing_config.data_preprocessing_artifacts_dir, exist_ok=True)


            metadata = os.path.join(self.data_ingestion_artifact.extracted_data_path, METADATA_FILE_NAME)

            waves_path = self.data_ingestion_artifact.extracted_data_path
            wavs = None
            logging.info("Writing the path to wavs")
            self.wavs = glob("{}/**/*.wav".format(waves_path), recursive=True)

            logging.info("Creating the dictionary to id_to_text")
            self.id_to_text = {}
            with open(metadata, encoding="utf-8") as f:
                for line in f:
                    id = line.strip().split("|")[0]
                    text = line.strip().split("|")[2]
                    self.id_to_text[id] = text
            
            os.makedirs(self.data_preprocessing_config.metadata_dir_path, exist_ok=True)

            
            with open(self.data_preprocessing_config.waves_file_path, 'w') as f:
                write = csv.writer(f)
                write.writerows(self.wavs)

            logging.info("Exiting the get_id_to_index method of DataPreprocessing")
            return self.wavs, self.id_to_text
        except Exception as e:
            raise STTException(e, sys)

    def extract_data(self) -> None:
        try:
            logging.info("Entering the extract_data method of preprocesing")
            self.data = get_data(self.wavs, self.id_to_text, maxlen=MAX_TARGET_LENGTH)

            logging.info("Exiting the extract_data method of preprocessing")
        except Exception as e:
            raise STTException(e, sys)
    
    def train_test_split(self) -> tuple:
        try:
            logging.info("Entered the train_test_split method of preprocessing")
            split = int(len(self.data) * TRAIN_TEST_SPLIT_RATIO)
            train_data = self.data[:split]
            test_data = self.data[split:]

            logging.info("write train data")
            os.makedirs(self.data_preprocessing_config.train_dir_path, exist_ok=True)

            keys = train_data[0].keys()
            self.train_file_path = os.path.join(self.data_preprocessing_config.train_dir_path, TRAIN_FILE_NAME)

            with open(self.train_file_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(train_data)


            logging.info("Write test data")
            os.makedirs(self.data_preprocessing_config.test_dir_path, exist_ok=True)

            self.test_file_path = os.path.join(self.data_preprocessing_config.test_dir_path, TEST_FILE_NAME)
            with open(self.test_file_path, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(test_data)
                output_file.close()
            
            logging.info("Train and test split of the data is done")
            return train_data, test_data
        except Exception as e:
            raise STTException(e, sys)


    def initiate_data_preprocessing(self):
        try:
            logging.info("Initiate data preprocessing...")
            self.get_id_to_text()
            self.extract_data()
            self.train_test_split()

            data_processing_artifact = DataPreprocessingArtifacts(
                train_data_path=self.train_file_path,
                test_data_path=self.test_file_path
            )
            logging.info("Data Preprocessing completed succesfully.")
            return data_processing_artifact
        except Exception as e:
            raise STTException(e, sys)