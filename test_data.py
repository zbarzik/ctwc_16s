#!/usr/bin/python
from common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP
import warnings

import numpy as np

def get_default_tree(otus):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    tree_str = "(((A:0.15)B:0.2,(C:0.3,D:0.4)E:0.6)F:0.1)G;"
    for i, otu in enumerate(otus):
        tree_str = tree_str.replace(chr(ord('A') + i), otu)
    tr = DndParser(tree_str, UniFracTreeNode)
    return tr

def get_sample_biom_table():
    table = get_biom_table_from_file('305_otu_table.json')
    #table = get_biom_table_from_file('486_otu_table.json')
    if table is not None:
        return table.matrix_data.todense(), table.ids('observation'), table.ids('sample')
    return None

def get_default_samples():
    return np.array(['mouth', 'leg', 'lips', 'armpit', 'foot'])

def get_default_otus():
    return np.array(['donatello', 'leonardo', 'raphael', 'michelangelo', 'splinter', 'shredder', 'april'])

def get_default_data(otus, samples):
    #return get_randomly_pre_generated_data()
    return get_synthetic_cluster_data(otus, samples)

def get_synthetic_cluster_data(otus, samples):
    data = get_randomly_pre_generated_data() # noise
    data += np.array([   [5, 13, 0, 0, 0],
                         [8, 15, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    return data

def get_randomly_generated_data(otus, samples):
    num_otu = len(otus)
    num_samp = len(samples)
    data = np.random.randint(low=0, high=2, size=(num_otu, num_samp))
    return data

def get_randomly_pre_generated_data():
    # For deterministic testing - this was generated once using the above function
    data = np.array([   [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [0, 0, 1, 1, 0],
                        [1, 1, 0, 1, 1],
                        [1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 0],
                        [0, 0, 0, 1, 1]])

    return data

def test():
    samples = get_default_samples()
    otus = get_default_otus()
    tree = get_default_tree(otus)
    data = get_default_data(otus, samples)
    DEBUG("Tree:\n" + tree.asciiArt())
    DEBUG("Samples:\n" + str(samples))
    DEBUG("OTUs:\n" + str(otus))
    DEBUG("Data:\n" + str(data))

if __name__ == "__main__":
    test()
