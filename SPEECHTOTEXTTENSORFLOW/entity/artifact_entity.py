from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    downloaded_data_path: str
    extracted_data_path: str

@dataclass
class DataPreprocessingArtifacts:
    train_data_path: str
    test_data_path: str

@dataclass
class ModelTrainerArtifacts:
    model_path: str
    model_loss: int

@dataclass
class ModelEvaluationArtifacts:
    s3_model_loss: float
    is_model_accepted: bool
    trained_model_path: str
    s3_model_path: str

@dataclass
class ModelPusherArtifacts:
    response: dict