#!/usr/bin/python

import logging, pdb
import cPickle, gzip
import numpy

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

def save_to_file(obj_to_save, filename, mat=None, mat_fn=None):
    try:
        fn = gzip.GzipFile(filename, 'wb+')
        try:
            cPickle.dump(obj_to_save, fn, -1)
        except Exception as ex:
            WARN("Error trying to write object to file {0}: {1}".format(filename, str(ex)))
        fn.close()
        if mat is not None and mat_fn is not None:
            try:
                with open(mat_fn, 'wb+') as fn:
                    numpy.savez_compressed(fn, mat)
            except Exception as ex:
                WARN("Error writing matrix to file {0}: {1}".format(mat_fn, str(ex)))

    except Exception as ex:
        WARN("Error writing to file {0}: {1}".format(filename, str(ex)))

def load_from_file(filename, with_mat=False, mat_fn=None):
    obj = None
    mat = None
    try:
        fn = gzip.GzipFile(filename, 'rb')
        try:
            obj = cPickle.load(fn)
        except Exception:
            DEBUG("Error trying to read object from file {0}".format(filename))
        fn.close()
    except Exception:
        DEBUG("Error trying to read file {0}".format(filename))
    if not with_mat:
        return obj
    try:
        with open(mat_fn, 'rb') as fn:
            mat = numpy.load(fn)
    except Exception as ex:
        DEBUG("Error trying to read mat_{0} file: {1}:".format(mat_fn, str(ex)))
    return obj, mat

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
