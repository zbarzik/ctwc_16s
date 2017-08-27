#!/usr/bin/python
import logging, pdb
import cPickle, gzip
import numpy
import bisect
from functools import wraps
from collections import namedtuple

LOG_LEVEL_CONSOLE = logging.INFO
LOG_LEVEL_FILE = logging.DEBUG

LOG_FILE = "ctwc__logger.log"
logger = None
MAX_PRINT_SIZE = 5000

TITLE_FORMAT = "Iteration {0}" # iteration is assumed to be at the end of the title format

CLUSTER_SUMMARY_CSV = "cluster_summary.csv"

def memoize(function):
    memo = {}
    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

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

def ASSERT(predicate):
    import traceback
    if not predicate:
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
            data = numpy.load(fn)
            mat = data['arr_0']
    except Exception as ex:
        DEBUG("Error trying to read mat_{0} file: {1}:".format(mat_fn, str(ex)))
    return obj, mat

def has_value(sorted_list, value):
    if sorted_list is None:
        return False

    if type(sorted_list) is set:
        return value in sorted_list # O(1) operation

    # Otherwise assuming the list is sorted, O(lg(n)) operation
    i = bisect.bisect_left(sorted_list, value)
    if i != len(sorted_list) and sorted_list[i] == value:
        return True
    return False

def make_camel_from_string(header):
    cameled_hdr = ''.join(x.title() for x in header.split())
    sanitized_hdr = ''.join(x for x in cameled_hdr if x.isalnum())
    return sanitized_hdr

def write_dict_as_csv(filename, mydict, ref_dict):
    import csv
    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key in mydict:
            valye = mydict[key]
            original_val = "na"
            if key in ref_dict:
                original_val = ref_dict[key]
            writer.writerow([key, (value, original_val)])

def write_dict_entry_to_open_csv_file(csv_writer, field, mydict, sel_dict, ref_dict, num_selected=0.0, num_total=0.0):
    for key in mydict:
        value = mydict[key]
        accuracy = 0.0
        distribution = 0.0
        if key in sel_dict:
            distribution = sel_dict[key]
        ref_distribution = 0.0
        if key in ref_dict:
            ref_distribution = ref_dict[key]
            if num_total > 0 and ref_distribution > 0:
                accuracy =  (distribution * num_selected) / (ref_distribution * num_total)
            else:
                WARN("Cannot calculate accuracy value for field={0}, key={1}: ref_distribution={2}, num_total={3}".format(field, key, ref_distribution, num_total))
        csv_writer.writerow([field, key, (value, distribution, ref_distribution, accuracy)])

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

def get_iteration_path_string(iteration_ind):
    def get_string_for_step(step):
        if step == "1":
            return "pick samples top cluster"
        elif step == "2":
            return "pick OTUs top cluster"
        elif step == "3":
            return "pick samples top cluster using samples compliment"
        elif step == "4":
            return "pick OTUs top cluster using OTUs compliment"
        elif step == "5":
            return "pick samples top cluster using OTUs compliment"
        elif step == "6":
            return "pick OTUs top cluster using samples compliment"
        else:
            FATAL("Ilegal string value")

    res = ""
    while len(iteration_ind) > 0:
        if len(res) > 0:
            res += "; "
        res += get_string_for_step(iteration_ind[0])
        if len(iteration_ind) > 2:
            iteration_ind = iteration_ind[2:]
        else:
            iteration_ind = ""

    return res[0].upper() + res[1:]

def iteration_to_title(iteration):
    return TITLE_FORMAT.format(iteration)

def title_to_iteration(title):
    pref = TITLE_FORMAT.format("")
    if not title.startswith(pref):
        FATAL("Ilegal title string")
    return title[len(pref):]

LineStructure_Fields = ['title',
                        'clustering_sequence',
                        'input_otus',
                        'input_samples',
                        'clustering_dimension',
                        'picked_cluster_size',
                        'label_1',
                        'value_1',
                        'q_val_1',
                        'purity_1',
                        'reference_abundance_1',
                        'accuracy_1',
                        'label_2',
                        'value_2',
                        'q_val_2',
                        'purity_2',
                        'reference_abundance_2',
                        'accuracy_2',
                        ]

LineStructure = namedtuple('LineStructure', LineStructure_Fields)

QValStructure_Fields = ['label',
                        'value',
                        'q_val',
                        'purity',
                        'ref_abundance',
                        'accuracy',
                        ]

QValStructure = namedtuple('QValStructure', QValStructure_Fields)

def parse_result_file_to_structured_line(filename):
    def is_interesting_q_val(qval, picked_size, axis):
        if qval.value.lower() == "na":
            return False
        elif qval.value.lower() == "missing":
            return False
        elif qval.value.lower() == "":
            return False
        elif round(picked_size * qval.purity) == 1.0:
            return False
        elif qval.purity < qval.ref_abundance * 2: # arbitrary...
            return False
        elif axis == 'samples' and qval.purity < 0.1:
            return False
        elif axis == 'samples' and qval.accuracy < 0.1:
            return False
        else:
            return True

    with open(filename, 'r') as fh:
        lines = fh.readlines()
        title = lines[0].split('-')[0].strip()
        clustering_sequence = get_iteration_path_string(title_to_iteration(title))
        input_otus = int(lines[0].split('-')[1].strip().split(' ')[0].strip())
        input_samples = int(lines[0].split('-')[1].strip().split(' ')[3].strip())
        if lines[2].startswith("---"):
            next_line = 3
        else:
            next_line = 2
        clustering_dimension = lines[next_line].split()[2][:-1]
        picked_cluster_size = int(lines[next_line].split()[1].strip())

        reached_q_vals = False
        reached_separator = False
        q_vals = []
        for s_line in lines:
            line = s_line.strip()
            if len(line) == 0:
                continue
            if line.lower().startswith("filtered q values"):
                reached_q_vals = True
                continue
            if reached_q_vals and not reached_separator:
                reached_separator = True
                continue
            if reached_q_vals:
                label = line.split(',')[0]
                value = line.split(',')[1]
                q_val = float(line.split(',')[2][2:])
                purity = float(line.split(',')[3])
                ref_abundance = float(line.split(',')[4])
                accuracy = float(line.split(',')[5][:-2])

                q_vals.append(QValStructure(label, value, q_val, purity, ref_abundance, accuracy))


        best = QValStructure("", "", 1.0, 0.0, 0.0, 0.0)
        second_best = QValStructure("", "", 1.0, 0.0, 0.0, 0.0)
        for qval in q_vals:
            if not is_interesting_q_val(qval, picked_cluster_size, clustering_dimension):
                continue
            if (qval.purity > best.purity or
                qval.purity == best.purity and qval.accuracy > best.accuracy):
                second_best = best
                best = qval
            elif (qval.purity > second_best.purity or
                  qval.purity == second_best.purity and qval.accuracy > second_best.accuracy):
                second_best = qval

        return LineStructure(title,
                             clustering_sequence,
                             input_otus,
                             input_samples,
                             clustering_dimension,
                             picked_cluster_size,
                             best.label,
                             best.value,
                             best.q_val,
                             best.purity,
                             best.ref_abundance,
                             best.accuracy,
                             second_best.label,
                             second_best.value,
                             second_best.q_val,
                             second_best.purity,
                             second_best.ref_abundance,
                             second_best.accuracy,
                            )

def write_cluster_summary_as_csv(output_filename, cluster_filename_list):
    import csv
    with open(output_filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(LineStructure_Fields)
        for fn in cluster_filename_list:
            writer.writerow(parse_result_file_to_structured_line(fn))

def write_cluster_summary_for_all_files_in_path():
    import glob
    write_cluster_summary_as_csv(CLUSTER_SUMMARY_CSV, glob.glob('cluster*.txt'))

init_logger()
