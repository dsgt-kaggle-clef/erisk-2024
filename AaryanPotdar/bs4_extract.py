from bs4 import BeautifulSoup
import os

folder_path_2023 = "/Users/aaryanpotdar/Desktop/eRisk 2024/task3/training/t3_training/TRAINING DATA (FROM ERISK 2022 AND 2023)/2023/erisk 2023_T3/eRisk2023_T3_Collection/"
folder_path_2022 = '/Users/aaryanpotdar/Desktop/eRisk 2024/task3/training/t3_training/TRAINING DATA (FROM ERISK 2022 AND 2023)/2022/T3 2022/eRisk2022_T3_Collection'
# output_dir = '/Users/aaryanpotdar/Desktop/eRisk 2024/2023_path'
output_dir = '/Users/aaryanpotdar/Desktop/eRisk 2024/2022_path'

# get list of files
xml_files = [file for file in os.listdir(folder_path_2022) if file.endswith('.xml')]

for f in xml_files:
    file_path  = os.path.join(folder_path_2022, f)

    with open(file_path, "r") as file:
        contents = file.read()

    soup = BeautifulSoup(contents, 'xml')

    root = soup.find('INDIVIDUAL')

    output_file_name = os.path.splitext(f)[0] + '.txt'
    output_file_path = os.path.join(output_dir, output_file_name)

    # Write the contents to the text file
    with open(output_file_path, 'w') as output_file:
        output_file.write(root.text)