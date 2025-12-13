import os
import sys
import numpy as np

from SPEECHTOTEXTTENSORFLOW.exceptions import STTException
from SPEECHTOTEXTTENSORFLOW.logger import logging
from SPEECHTOTEXTTENSORFLOW.cloud_storage.s3_operations import S3Sync
from SPEECHTOTEXTTENSORFLOW.entity.config_entity import ModelEvaluationConfig
from SPEECHTOTEXTTENSORFLOW.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts
from SPEECHTOTEXTTENSORFLOW.constants import MAX_TARGET_LENGTH
from SPEECHTOTEXTTENSORFLOW.models.model import Transformer


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifacts):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise STTException(e, sys)
    
    def get_best_model_path(self):
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            os.makedirs(self.model_evaluation_config.model_evaluation_artifact_dir, exist_ok=True)
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(folder=best_model_dir, aws_bucket_url=model_path)
            if len(os.listdir(best_model_dir)) != 0:
                best_model_path = best_model_dir
                logging.info(f"Best model found in {best_model_path}")
            else:
                logging.info("Model is not available in best_model_directory")
            
            return best_model_path
        except Exception as e:
            raise STTException(e,sys)

    def evaluate_model(self):
        try:
            best_model_path = self.get_best_model_path()
            if best_model_path is not None:
                s3_model = Transformer(
                    num_hid=200,
                    num_head=2,
                    num_feed_forward=400,
                    target_maxlen=MAX_TARGET_LENGTH,
                    num_layers_enc=4,
                    num_layers_dec=1,
                    num_classes=34,
                )
                s3_model.built = True
                s3_model.load_weights(best_model_path)

                val_loss = s3_model.val_loss.numpy()
                logging.info(f"S3 Model Validation loss is {val_loss}")
                logging.info(f"Locally trained accuracy is {self.model_trainer_artifact.model_loss}")

                s3_model_loss = val_loss
            else:
                logging.info("Model is not found on production server, So couldn't evaluate")
                s3_model_loss = None
            return s3_model_loss
        except Exception as e:
            raise STTException(e, sys)
    
    def initiate_model_evaluation(self):
        try:
            
            s3_model_loss = self.evaluate_model()
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss

            trained_model_loss = self.model_trainer_artifact.model_loss

            evaluation_response = tmp_best_model_loss > trained_model_loss

            model_evaluation_artifacts = ModelEvaluationArtifacts(s3_model_loss = s3_model_loss,
                                                                is_model_accepted = evaluation_response,
                                                                trained_model_path = self.model_trainer_artifact.model_path,
                                                                s3_model_path = self.get_best_model_path()
                                                                )
            logging.info(f"Model evaludation completed! Artifacts: {model_evaluation_artifacts}")

            return model_evaluation_artifacts
        except Exception as e:
            raise STTException(e, sys)