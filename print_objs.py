import json


with open('val2014_objs.json', 'r') as f:
    data=json.load(f)

print(data[0].keys())