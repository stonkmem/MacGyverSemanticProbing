import json
import sys

def read_txt_file_as_json(txt_file_path):
    with open(txt_file_path, 'r') as txt_file:
        data = json.load(txt_file)
    return data

def read_json_file(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

file_name = "results.json"
if len(sys.argv) > 1:
    file_name = sys.argv[1]
# Example use case
data = read_json_file(file_name)
print(data)