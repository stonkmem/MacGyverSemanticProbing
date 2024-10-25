import json

def read_txt_file_as_json(txt_file_path):
    with open(txt_file_path, 'r') as txt_file:
        data = json.load(txt_file)
    return data

# Example use case
file_contents = read_txt_file_as_json("outputs.txt")
print(file_contents)