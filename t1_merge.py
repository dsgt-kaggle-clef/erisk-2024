### This scripts reads the TREC Files and compiles them into one file ###

import pandas as pd
import glob
import ntpath
from pathlib import Path


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# folder_path = "./task1_test"

# path =r'.\task1\test' 
path =r'.\task1_test' 
# Only trec files
allFiles = glob.glob(path + "/*.trec")

# i = 0

# for file_ in allFiles:
#     print(i)
#     i = i+1


with open('./merge.trec', 'w') as outfile:
    for file_ in allFiles:
        with open(file_) as infile:
            outfile.write(infile.read())





