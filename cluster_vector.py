#!/usr/bin/python
from common import DEBUG,INFO,WARN,ERROR,FATAL

from optparse import OptionParser
parser = OptionParser()
try:
    parser.add_option("-m", "--metric", action="store", dest="metric_function_name", help="Metric function used to compare cells")
except TypeError as te:
    FATAL("Failed adding option %s to parser" %str(te))
except:
    FATAL("Failed adding option to parser(unknown exception)")


if __name__ == '__main__':
    pass
