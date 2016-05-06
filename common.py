#!/usr/bin/python

import logging, pdb

LOG_LEVEL_CONSOLE = logging.DEBUG
LOG_LEVEL_FILE = logging.DEBUG

LOG_FILE = "ctwc_logger.log"
logger = None
MAX_PRINT_SIZE = 5000

def DEBUG(message):
    if len(str(message)) > MAX_PRINT_SIZE:
        logger.debug(message[:MAX_PRINT_SIZE/2] + "\n(>>>)\n" + message[-MAX_PRINT_SIZE/2:])
    else:
        logger.debug(message)

def ERROR(message):
    logger.error(message)

def INFO(message):
    logger.info(message)

def WARN(message):
    logger.warn(message)

def FATAL(message):
    ERROR(message)
    exit(-1)

def ASSERT(condition):
    import traceback
    if not condition:
        traceback.print_stack()
        ERROR("Assertion failed!")
        BP()

def BP():
    pdb.set_trace()

def init_logger():
    file_handler = logging.FileHandler(filename=LOG_FILE, mode='a')
    console_handler = logging.StreamHandler()
    file_handler.setLevel(LOG_LEVEL_FILE)
    console_handler.setLevel(LOG_LEVEL_CONSOLE)
    globals()['logger'] = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


init_logger()
