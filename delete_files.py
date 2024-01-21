import os

def delete_json_files(directory):
    # Traverse through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Check if the file is a json file
            if filename.endswith('.json'):
                # Get the full path of the file
                file_path = os.path.join(dirpath, filename)
                # Delete the file
                os.remove(file_path)
                print(f"Deleted {file_path}")

# Specify the directory you want to start from
start_dir = 'Papers'
delete_json_files(start_dir)
