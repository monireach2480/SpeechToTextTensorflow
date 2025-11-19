from SPEECHTOTEXTTENSORFLOW.entity.config_entity import DataIngestionConfig
from SPEECHTOTEXTTENSORFLOW.components.data_ingestion import DataIngestion

di_ins = DataIngestion(DataIngestionConfig)
di_art = di_ins.initiate_data_ingestion()