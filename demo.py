# demo.py
from SPEECHTOTEXTTENSORFLOW.entity.config_entity import (
    DataIngestionConfig, DataPreprocessingConfig, ModelTrainerConfig
)
from SPEECHTOTEXTTENSORFLOW.components.data_ingestion import DataIngestion
from SPEECHTOTEXTTENSORFLOW.components.data_preprocessing import DataPreprocessing
from SPEECHTOTEXTTENSORFLOW.components.model_trainer import ModelTrainer

# Step 1
di_config = DataIngestionConfig()
di = DataIngestion(di_config)
di_artifact = di.initiate_data_ingestion()

# Step 2
dp_config = DataPreprocessingConfig()
dp = DataPreprocessing(dp_config, di_artifact)
dp_artifact = dp.initiate_data_preprocessing()

# Step 3
trainer_config = ModelTrainerConfig()
trainer = ModelTrainer(dp_artifact, trainer_config)
trained_artifact = trainer.initiate_model_trainer()

print("\nSUCCESS! Model trained and saved at:")
print(trained_artifact.model_path)
print(f"Final validation loss: {trained_artifact.model_loss:.4f}")