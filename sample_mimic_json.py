import json
# manually load json file
ann_path = 'data/mimic_cxr/annotation.json'  
mimic_ann = json.loads(open(ann_path, 'r').read())
# print(mimic_ann.keys())
# print('the number of training images', len(mimic_ann['train']))
# print('the number of testing images', len(mimic_ann['test']))
# print('the number of validating images', len(mimic_ann['val']))
# the number of training images 270790
# the number of testing images 3858
# the number of validating images 2130
# sample_length = 10000
sample_length = 100
sampled_train = mimic_ann['train'][0:sample_length]
sampled_test = mimic_ann['test'][0:10]
sampled_val =  mimic_ann['val'][2000:]
print('the length of sampled_train', len(sampled_train))
sampled_mimic_ann = {}
sampled_mimic_ann['train'] = sampled_train
sampled_mimic_ann['val'] = sampled_val
sampled_mimic_ann['test'] = sampled_test
# with open('data/mimic_100/annotation100.json', 'w', encoding='utf-8') as f:
# with open('data/mimic_100/annotation_sample.json', 'w', encoding='utf-8') as f:
with open('data/mimic_100/annotation_db3.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_mimic_ann, f, ensure_ascii=False, indent=4)