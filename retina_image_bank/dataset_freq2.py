import os
import json
import random
import shutil
import csv
import collections

cleaned_report_path = '/media/hdd/data/imcaption/danli_datav2/cleaned.csv'
img_folder_path = '/media/hdd/data/imcaption/danli_datav2/images819'
cleaned_csv_list = []
main_disease_list = []
sub_disease_list = []
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
            main_disease_list.append(single_dic['DS'])
            sub_disease_list.append(single_dic['Ds'])
# print(cleaned_csv_list)
main_disease_dict = collections.Counter(main_disease_list)
sub_disease_dict = collections.Counter(sub_disease_list)
# print('disease_dict', disease_dict)
# print(len(disease_dict))
main_disease_set = set()
sub_disease_set = set()
for diesease in main_disease_dict.keys():
    disease_text = diesease.split(",")
    # print(disease_text)
    for eye_disease in disease_text:
        main_disease_set.add(eye_disease)

for diesease in sub_disease_dict.keys():
    disease_text = diesease.split(",")
    # print(disease_text)
    for eye_disease in disease_text:
        sub_disease_set.add(eye_disease)
# print(disease_set)
combine_set = main_disease_set | sub_disease_set
print(len(combine_set))