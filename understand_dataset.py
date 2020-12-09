import json
file_name = 'data/iu_xray/annotation.json'
with open(file_name) as json_file:
    data = json.load(json_file)
    keys = data.keys()
print(keys)
print('the number of reports for testing', len(data['test']))
print('the number of reports for training', len(data['train']))
print('the number of reports for validating', len(data['val']))

file_name = 'data/mimic_cxr/annotation.json'
with open(file_name) as json_file:
    data = json.load(json_file)
    keys = data.keys()
print(data['test'][0])
print('the number of reports for testing', len(data['test']))
print('the number of reports for training', len(data['train']))
print('the number of reports for validating', len(data['val']))
