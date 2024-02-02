### This script reads the merged TREC Files and converts into a PD dataframe then parquet ###

import xml.etree.ElementTree as ElementTree
import pandas as pd

with open('merge.trec', 'r', encoding="utf8") as f:   # Reading file
    xml = f.read()

xml = xml.replace('&<','&amp;<')

xml = '<ROOT>' + xml + '</ROOT>'   # Let's add a root tag

# root = ElementTree.fromstring(xml)

df = pd.read_xml(xml)

print(df.head())


