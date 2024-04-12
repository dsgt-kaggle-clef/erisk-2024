import os
import io
from lxml import etree
import json
import re

#folder paths
folder_path = "/Users/aaryanpotdar/Desktop/Kaggle_CLEF/task3/training/t3_training/TRAINING DATA (FROM ERISK 2022 AND 2023)/2023/erisk 2023_T3/eRisk2023_T3_Collection"
# folder_path = '/Users/aaryanpotdar/Desktop/Kaggle_CLEF/task3/training/t3_training/TRAINING DATA (FROM ERISK 2022 AND 2023)/2022/T3 2022/eRisk2022_T3_Collection'
output_folder = "/Users/aaryanpotdar/Desktop/Kaggle_CLEF/json_files/2023_path"
# output_folder = "/Users/aaryanpotdar/Desktop/Kaggle_CLEF/json_files/2022_path"


# Function to escape < and > in text nodes
def clean_xml_text(node):
    if node.text:
        node.text = node.text.replace('<', '&lt;').replace('>', '&gt;')
    for child in node:
        clean_xml_text(child)

# use function only if needed to parse & character
def clean_ampersand(xml_text):
    # replace & with &amp, only if not part of an entity already
    xml_text = re.sub(r'&(?!(amp|lt|gt|quot|apos);)', '&amp;', xml_text)
    return xml_text

# create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.endswith('.xml'):
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        
            parser = etree.XMLParser(recover=True)
            tree = etree.fromstring(content, parser=parser)
            clean_xml_text(tree)

            content = etree.tostring(tree, encoding='unicode')
        
            tree = etree.parse(io.StringIO(content))
            root = tree.getroot()

            json_data = {
                "ID": root.find('ID').text,
                "posts": []
            }

            for writing in root.findall('WRITING'):
                post_data = {
                    "title": writing.find('TITLE').text.strip(),
                    "date": writing.find('DATE').text.strip(),
                    "info": writing.find('INFO').text.strip(),
                    "text": writing.find('TEXT').text.strip()
                }
                json_data["posts"].append(post_data)

            json_output = json.dumps(json_data, indent=4)

            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, 'w') as f:
                f.write(json_output)

        except etree.XMLSyntaxError as e:
            print(f"Could not parse the file {file_path}: due to error {e}")
            
print("xml_files parsed")