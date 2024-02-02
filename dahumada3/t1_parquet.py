### This scripts reads the TREC Files and outputs parquet files ###

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

# Path(file_).stem

# with open('./merge.trec', 'w') as outfile:
#     for file_ in allFiles:
#         with open(file_) as infile:
#             outfile.write(infile.read())

for file_ in allFiles:
    with open(file_, 'r', encoding="utf8") as f:
        xml = f.read()   # Reading file
        xml = xml.replace('&<','&amp;<')
        xml = '<ROOT>' + xml + '</ROOT>'   # Let's add a root tag           
        df = pd.read_xml(xml)
        filename = Path(file_).stem
        df.to_parquet(".\parquet\\"+filename+".parquet")
            

