import os
import re

def rename_files(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            old_file_path = os.path.join(dirpath, filename)
            new_filename = re.sub(r'^\d+\.\d+v\d+_', '', filename)
            new_filename = new_filename[new_filename.find(' ') + 1:]

            new_filename = re.sub(r'_', ' ', new_filename)
            new_file_path = os.path.join(dirpath, new_filename)
            print(new_file_path)
            os.rename(old_file_path, new_file_path)


rename_files('Papers/2024')
