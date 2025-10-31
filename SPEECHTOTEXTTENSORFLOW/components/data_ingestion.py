import sys
import os
from zipfile import ZipFile

from SPEECHTOTEXTTENSORFLOW.exceptions import STTException
from SPEECHTOTEXTTENSORFLOW.logger import logging

from SPEECHTOTEXTTENSORFLOW.entity.config_entity import DataIngestionConfig
from SPEECHTOTEXTTENSORFLOW.entity.artifact_entity import DataIngestionArtifacts
from SPEECHTOTEXTTENSORFLOW.constants import *
from SPEECHTOTEXTTENSORFLOW.cloud_storage.s3_operations import S3Sync


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
        except Exception as e:
            raise STTException(e, sys)
        
    def get_data_from_cloud(self) -> None:
        try:
            logging.info("Initiating data download from s3 bucket...")
            download_dir = self.data_ingestion_config.download_dir
            bucket_uri = self.data_ingestion_config.bucket_uri
            # s3_zip_file_path = self.data_ingestion_config.s3_zip_file_path
            if os.path.isdir(download_dir):
                logging.info(
                    f"Data is already present in {download_dir}, So skipping download step.")
                return None
            else:
                os.makedirs(download_dir, exist_ok=True)
                self.s3_sync.sync_folder_from_s3(download_dir, bucket_uri)
                logging.info(f"Data is downloaded from s3 bucket to Download directory: {download_dir}.")
        except Exception as e:
            raise STTException(e, sys)
    
    def unzip_data(self) -> None:
        try:
            logging.info("Unzipping the downloaded zip file from download directory...")
            s3_zip_file_path = self.data_ingestion_config.s3_zip_file_path
            unzip_data_dir_path = self.data_ingestion_config.unzip_data_dir_path
            unzip_data_dir = os.path.join(unzip_data_dir_path, UNZIPPED_FOLDER_NAME)
            if os.path.isdir(unzip_data_dir):
                logging.info(
                    "Unzipped Folder already exists in unzip directory, so skipping unzip operation.")
            else:
                os.makedirs(unzip_data_dir_path, exist_ok=True)
                with ZipFile(s3_zip_file_path, 'r') as zip_file_ref:
                    zip_file_ref.extractall(unzip_data_dir_path)
                logging.info(
                    f"Unzipped file exists in unzip directory: {unzip_data_dir_path}.")
        except Exception as e:
            raise STTException(e, sys)

    
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("Initiating the data ingestion component...")
            self.get_data_from_cloud()
            self.unzip_data()

            extracted_data_path = os.path.join(self.data_ingestion_config.unzip_data_dir_path, UNZIPPED_FOLDER_NAME)


            data_ingestion_artifact = DataIngestionArtifacts(
                downloaded_data_path=self.data_ingestion_config.download_dir,
                extracted_data_path=extracted_data_path
            )
            logging.info('Data ingestion is completed Successfully.')

            return data_ingestion_artifact
        except Exception as e:
            raise STTException(e, sys)