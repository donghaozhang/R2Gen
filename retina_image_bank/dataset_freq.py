import os
import json
import random
import shutil
import csv
import collections

random.seed(10)
json_path = '/media/hdd/donghao/imcaption/R2Gen/data/danli_datav2/annotationv2_debug.json'
with open(json_path) as f:
  json_data = json.load(f)
# print(json_data.keys())
# # dataset_split = 'train'
# dataset_split = 'test'
print('the length of training set', len(json_data['train']))
# print('train sample', json_data['train'][0])
print('the length of testing set', len(json_data['test']))
# print('test sample', json_data['test'][0])
print('the length of validatiing set', len(json_data['val']))

main_disease_flag = False
sub_disease_flag = True
cleaned_report_path = '/media/hdd/data/imcaption/danli_datav2/cleaned.csv'
img_folder_path = '/media/hdd/data/imcaption/danli_datav2/images819'
cleaned_csv_list = []
disease_list = []
with open(cleaned_report_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        single_dic = {}
        if line_count == 0:
            column_title = row
            line_count += 1
        else:
            single_dic['id'] = row[0]
            single_dic['text'] = row[1]
            single_dic['pred_modality'] = row[2]
            single_dic['DS'] = row[3]
            single_dic['Ds'] = row[4]
            single_dic['LS'] = row[5]
            single_dic['ls'] = row[6]
            single_dic['dir'] = os.path.join(img_folder_path, os.path.basename(row[7]))
            single_dic['caption'] = row[8]
            cleaned_csv_list.append(single_dic)
            line_count += 1
            # print('The current path is', single_dic['dir'])
            # print(single_dic['DS'])
            if main_disease_flag:
                disease_list.append(single_dic['DS'])
            elif sub_disease_flag:
                disease_list.append(single_dic['Ds'])
# print(cleaned_csv_list)
disease_dict = collections.Counter(disease_list)
# print('disease_dict', disease_dict)
# print(len(disease_dict))
disease_set = set()
for diesease in disease_dict.keys():
    disease_text = diesease.split(",")
    # print(disease_text)
    for eye_disease in disease_text:
        disease_set.add(eye_disease)
# print(disease_set)
disease_set.remove('')
# print(disease_set)
disease_cnt = {}
for disease_name in disease_set:
	disease_cnt[disease_name] = 0
with open(cleaned_report_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        single_dic = {}
        if line_count == 0:
            column_title = row
            line_count += 1
        else:
            single_dic['caption'] = row[8]
            caption = row[8]
            for disease_name in disease_set:
            	if disease_name in caption:
            		disease_cnt[disease_name] = disease_cnt[disease_name] + 1 
# print(disease_cnt)
sorted_dict_key = sorted(disease_cnt.items(), key = lambda kv: kv[1])
# print(sorted_dict_key)
total_disease = len(sorted_dict_key)
# print(total_disease)
print('total number of main disease', total_disease)
interval = int(total_disease/3)
low_freq = sorted_dict_key[0:interval]
middle_freq = sorted_dict_key[interval:interval*2]
high_freq = sorted_dict_key[interval*2:]
# print(high_freq)
# print(middle_freq)
# print(low_freq)
low_freq_name = []
middle_freq_name = []
high_freq_name = []
for cur in low_freq:
    low_freq_name.append(cur[0])
for cur in middle_freq:
    middle_freq_name.append(cur[0])
for cur in high_freq:
    high_freq_name.append(cur[0])
print(low_freq_name)
print(middle_freq_name)
print(high_freq_name)
