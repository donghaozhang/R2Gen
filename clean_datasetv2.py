import csv
import os
import json
from random import random
import re
from collections import Counter

cleaned_report_path = '/media/hdd/data/imcaption/danli_datav2/cleaned.csv'
img_folder_path = '/media/hdd/data/imcaption/danli_datav2/images819'
cleaned_csv_list = []
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
    print(f'Processed {line_count} lines.')
print('cur_dict[dir]', single_dic['dir'])
print(single_dic)

without_list = ['without history of trauma']
print('the length of dictionary', len(cleaned_csv_list))
without_counter = 0
for single_dic in cleaned_csv_list:
    if 'without' in  single_dic['text'] and len(single_dic['DS']) > 1:
        without_counter = without_counter + 1
        print(single_dic['id'], '!!!!!!!!')
        # print(single_dic['text'])
        word_index = re.search(r'\b(without)\b', single_dic['text'])
        start_value = word_index.start()
        print('start_value', start_value, single_dic['text'][start_value:start_value+30])
        print('DS', single_dic['DS'])
print('the number of without', without_counter)