#!/usr/bin/python

import logging, pdb

LOG_LEVEL_CONSOLE = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG

LOG_FILE = "ctwc__logger.log"
logger = None
MAX_PRINT_SIZE = 5000

def DEBUG(message):
    message = str(message)
    if len(message) > MAX_PRINT_SIZE:
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
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=LOG_FILE, mode='a')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    file_handler.setLevel(LOG_LEVEL_FILE)
    console_handler.setLevel(LOG_LEVEL_CONSOLE)
    globals()['logger'] = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)


init_logger()
