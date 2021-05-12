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
# print('cur_dict[dir]', single_dic['dir'])
# print(single_dic)

without_list = []
# print('the length of dictionary', len(cleaned_csv_list))
without_counter = 0
# the following code save sentences with the word "without" into a list
for single_dic in cleaned_csv_list:
    if 'without' in  single_dic['text'] and len(single_dic['DS']) > 1:
        without_counter = without_counter + 1
        # print(single_dic['id'], '!!!!!!!!')
        # print(single_dic['text'])
        word_index = re.search(r'\b(without)\b', single_dic['text'])
        start_value = word_index.start()
        # print('start_value', start_value, single_dic['text'][start_value:start_value+30])
        # print('DS', single_dic['DS'])
        without_list.append(single_dic['text'][start_value:start_value+30])
        # if 'cell or flare.' in single_dic['text']:
        #     print(single_dic['text'], '!!!!!!')
        #     print('******')
        #     print(single_dic['DS'])
        # if 'without known cause' in single_dic['text']:
        #     print(single_dic['text'], '!!!!!!')
        #     print('******')
        #     print(single_dic['DS'])
print('the number of without', without_counter)
# formulate the list into a dictionary
without_cnt = Counter(without_list)
# print(without_cnt)

# for key, value in sorted(without_cnt.items(), key=lambda item:item[1], reverse=True):
#     print(key, value)
# manually checking whether the word is safe or not
without_safe_list = ['without success.',
 'without therapy.', 'without any care before.',
  'without any problems.', 'without treatment', 'without cnvm', 'without known', 'without treatment']
print('the length of without dictionary before removing words', len(without_cnt))
without_cnt_copy = without_cnt.copy()
for key, value in without_cnt.items():
    for safe_item in without_safe_list:
        print_flag = False
        if safe_item in key:
            # print(key)
            del(without_cnt_copy[key])
            print_flag = False
        else:
            print_flag = True
    if print_flag:
        print(key, value)
print('the length of without dictionary after removing words', len(without_cnt_copy))
# {k: v for k, v in sorted(without_cnt.items(), key=lambda item: item[1])}without_safe_list