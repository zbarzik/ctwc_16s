#!/usr/bin/python

import logging

LOG_LEVEL_CONSOLE = logging.WARN
LOG_LEVEL_FILE = logging.DEBUG

LOG_FILE = "ctwc_logger.log"
logger = None

def DEBUG(message):
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
