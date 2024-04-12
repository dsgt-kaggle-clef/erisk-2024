import os
import xml.etree.ElementTree as ET

data_path = "/Users/aaryanpotdar/Desktop/eRisk 2024/task3/training/t3_training/TRAINING DATA (FROM ERISK 2022 AND 2023)/2023/erisk 2023_T3/eRisk2023_T3_Collection/"
file1 = "eRisk2023-T3_Subject1.xml"

file_path = data_path + file1
# Parse XML file into element 

# parser = ET.XMLParser(encoding='utf-8')
tree = ET.parse(file_path)
root = tree.getroot()
# root is the  INDIVIDUAL tag

print(root)
print(len(root))