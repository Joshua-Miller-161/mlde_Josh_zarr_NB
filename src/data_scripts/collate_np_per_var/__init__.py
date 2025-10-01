import sys
sys.dont_write_bytecode=True
import logging

logger = logging.getLogger(__name__)

from ...utils import input_to_list
#====================================================================
def get_variables(config):
    print(" >> >> INSIDE get_variables dataset_name", config.data.dataset_name)
    logger.info(" >> >> INSIDE get_variables dataset_name %s", config.data.dataset_name)
    
    variables = config.data.predictors.variables
    target_variables = config.data.predictands.variables

    return input_to_list(variables), input_to_list(target_variables)