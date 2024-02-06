from docClassify.pipeline.data_ingestion_step1 import DataIngestionPipeline
from docClassify.pipeline.data_validation_step2 import DataValidationPipeline
from docClassify.pipeline.data_preparation_step3 import DataPreparationPipeline
from docClassify.pipeline.data_train_and_validation_step4 import DataTrainValidationPipeline
from docClassify.logger import logger


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_ingestion = DataIngestionPipeline()
   #data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_validation = DataValidationPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data Preparation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_transformation = DataPreparationPipeline()
   #data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Training stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = DataTrainValidationPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
