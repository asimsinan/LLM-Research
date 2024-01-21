import json
import glob
files = glob.glob('**/000_Paper_List.json', recursive=True)
combined_contents = []

for file in files:
    with open(file, 'r') as f:
        contents = json.load(f)
        combined_contents.append(contents)

with open('Papers.json', 'w') as f:
    json.dump(combined_contents, f)