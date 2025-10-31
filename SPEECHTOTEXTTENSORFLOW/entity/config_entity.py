import os
from STT.constants import *

from datetime import datetime
from dataclasses import dataclass

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ROOT = os.getcwd()

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ROOT, ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    bucket_uri = DATA_BUCKET_URI
    file_name = ZIP_FILE_NAME
    download_dir = os.path.join(ROOT, DATA_DIR_NAME, DOWNLOAD_DIR)
    s3_zip_file_path: str = os.path.join(download_dir, file_name)
    unzip_data_dir_path: str = os.path.join(ROOT,DATA_DIR_NAME)

@dataclass
class DataPreprocessingConfig:
    data_preprocessing_artifacts_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_ARTIFACTS_DIR)
    metadata_dir_path = os.path.join(data_preprocessing_artifacts_dir, METADATA_DIR)
    waves_file_path: str = os.path.join(metadata_dir_path, WAVS_FILE_PATH)
    train_dir_path: str = os.path.join(data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TRAIN_DIR)
    test_dir_path: str = os.path.join(data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TEST_DIR)


@dataclass
class ModelTrainerConfig:
    model_dir_path = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_ARTIFACT_DIR)

@dataclass
class ModelEvaluationConfig:
    s3_model_path: str = S3_BUCKET_URI
    model_evaluation_artifact_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_ARTIFACT_DIR)
    best_model_dir: str = os.path.join(model_evaluation_artifact_dir, S3_MODEL_DIR_NAME)


@dataclass
class ModelPusherConfig:
    s3_model_path: str = S3_BUCKET_URI
    model_pusher_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR)

@dataclass
class PredictionPipelineConfig:
    s3_model_path: str = S3_BUCKET_URI
    prediction_artifact_dir = os.path.join(ROOT, PREDICTION_PIPELINE_DIR_NAME)
    model_download_path = os.path.join(prediction_artifact_dir, PREDICTION_MODEL_DIR_NAME)
    app_artifacts = os.path.join(prediction_artifact_dir, APPLICATION_ARTIFACTS_DIR)
    # input_sounds_path = os.path.join(app_artifacts, 'inputSound.mp3')
    wave_sounds_path = os.path.join(app_artifacts, 'input-wave.wav')