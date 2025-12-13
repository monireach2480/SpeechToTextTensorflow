
import os, sys

from SPEECHTOTEXTTENSORFLOW.components.data_ingestion import DataIngestion
from SPEECHTOTEXTTENSORFLOW.components.data_preprocessing import DataPreprocessing
from SPEECHTOTEXTTENSORFLOW.components.model_trainer import ModelTrainer
from SPEECHTOTEXTTENSORFLOW.components.model_evaluation import ModelEvaluation
from SPEECHTOTEXTTENSORFLOW.components.model_pusher import ModelPusher
from SPEECHTOTEXTTENSORFLOW.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from SPEECHTOTEXTTENSORFLOW.entity.artifact_entity import DataIngestionArtifacts, DataPreprocessingArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts, ModelPusherArtifacts
from SPEECHTOTEXTTENSORFLOW.logger import logging
from SPEECHTOTEXTTENSORFLOW.exceptions import STTException


class TrainingPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try: 
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion step completed successfully in train pipeline")
            return data_ingestion_artifacts
        except Exception as e:
            raise STTException(e, sys)
    
    def start_data_preprocessing(self, data_ingestion_artifacts: DataIngestionArtifacts) -> DataPreprocessingArtifacts:
        logging.info("Starting data preprocessing in training pipeline")
        try: 
            data_preprocessing = DataPreprocessing(data_preprocessing_config=self.data_preprocessing_config, data_ingestion_artifact=data_ingestion_artifacts)
            data_preprocessing_artifacts = data_preprocessing.initiate_data_preprocessing()
            logging.info("Data preprocessing step completed successfully in train pipeline")
            return data_preprocessing_artifacts
        except Exception as e:
            raise STTException(e, sys)
        
    def start_model_training(self, data_preprocessing_artifacts: DataPreprocessingArtifacts) -> ModelTrainerArtifacts:
        logging.info("Starting model training in training pipeline")
        try: 
            logging.info("Instantiating train and validation dataset from custom dataset class...")

            model_trainer = ModelTrainer(model_trainer_config=self.model_trainer_config, data_preprocessing_artifacats=data_preprocessing_artifacts)

            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Model trainer step completed successfully in train pipeline")
            return model_trainer_artifacts
        except Exception as e:
            raise STTException(e, sys)
    
    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Starting model evaluation in training pipeline")
        try: 
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                                model_trainer_artifact=model_trainer_artifacts)
            logging.info("Evaluating current trained model")
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation step completed successfully in train pipeline")
            return model_evaluation_artifacts
        except Exception as e:
            raise STTException(e, sys)

    def start_model_pusher(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        logging.info("Starting model pusher in training pipeline")
        try: 
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config, model_evaluation_artifacts=model_evaluation_artifacts)
            logging.info("If model is accepted in model evaluation. Pushing the model into production storage")
            model_pusher_artifacts = model_pusher.initiate_model_pusher()
            logging.info("Model pusher step completed successfully in train pipeline")
            return model_pusher_artifacts
        except Exception as e:
            raise STTException(e, sys)
    
    def run_pipeline(self) -> None:
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_preprocessing_artifacts = self.start_data_preprocessing(data_ingestion_artifacts=data_ingestion_artifacts)

            model_trainer_artifacts = self.start_model_training(data_preprocessing_artifacts=data_preprocessing_artifacts)

            model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts)
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifacts=model_evaluation_artifacts)

            logging.info("<<<< Training pipeline completed >>>>")
        except Exception as e:
            raise STTException(e, sys)