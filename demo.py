from SPEECHTOTEXTTENSORFLOW.entity.config_entity import DataIngestionConfig, DataPreprocessingConfig
from SPEECHTOTEXTTENSORFLOW.components.data_ingestion import DataIngestion
from SPEECHTOTEXTTENSORFLOW.components.data_preprocessing import DataPreprocessing

di_ins = DataIngestion(DataIngestionConfig)
di_art = di_ins.initiate_data_ingestion()

dp_ins = DataPreprocessing(DataPreprocessingConfig, di_art)
dp_art = dp_ins.initiate_data_preprocessing()