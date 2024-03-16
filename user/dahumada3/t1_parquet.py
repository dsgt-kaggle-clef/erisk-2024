### This scripts reads the TREC Files and outputs parquet files ###

import pandas as pd
import glob
from pathlib import Path
import unicodedata, re, itertools, sys


all_chars = (chr(i) for i in range(sys.maxunicode))
categories = {'Cc'}
# control_chars = ''.join(c for c in all_chars if unicodedata.category(c) in categories)
# or equivalently and much more efficiently
control_chars = ''.join(map(chr, itertools.chain(range(0x00,0x20), range(0x7f,0xa0))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def remove_control_chars(s):
    return control_char_re.sub('', s)

def trec2parquet():
    debugFlag = True

    # folder_path = "./task1_test"

    # path =r'.\task1\test' 
    path =r'.\task1\test' 
    # Only trec files
    allFiles = glob.glob(path + "/*.trec")

    # Path(file_).stem

    # with open('./merge.trec', 'w') as outfile:
    #     for file_ in allFiles:
    #         with open(file_) as infile:
    #             outfile.write(infile.read())

    for file_ in allFiles:
        with open(file_, 'r', encoding="utf8") as f:

            if debugFlag:
                print(file_)

            xml = f.read()   # Reading file
            xml = xml.replace('&<','&amp;<')
            xml = remove_control_chars(xml)
            xml = '<ROOT>' + xml + '</ROOT>'   # Let's add a root tag           
            df = pd.read_xml(xml)
            filename = Path(file_).stem
            df.to_parquet(".\parquet\\"+filename+".parquet")

if __name__ == "__main__":

    trec2parquet()