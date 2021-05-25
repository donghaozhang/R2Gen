import csv
import os
import json
from random import random
import re
from collections import Counter

# csv report path 
cleaned_report_path = '/media/hdd/data/imcaption/danli_datav2/cleaned.csv'
# image folder path
img_folder_path = '/media/hdd/data/imcaption/danli_datav2/images819'
cleaned_csv_list = []
# 1 open csv file 2 iterate each row of csv file 3 save information of a single line 
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

without_list = []
# print('the length of dictionary', len(cleaned_csv_list))
without_counter = 0
# the following code save sentences with the word "without" into a list
for single_dic in cleaned_csv_list:
    if 'without' in  single_dic['text'] and len(single_dic['DS']) > 1:
        without_counter = without_counter + 1
        word_index = re.search(r'\b(without)\b', single_dic['text'])
        start_value = word_index.start()
        without_list.append(single_dic['text'][start_value:start_value+30])
        # if 'without any shadowing' in single_dic['text']:
        #     print(single_dic['text'], '!!!!!!')
        #     print('******')
        #     print(single_dic['DS'])
print('the number of without', without_counter)
# formulate the without list into a dictionary and count the apperances 
without_cnt = Counter(without_list)

# manually checking whether the word is safe or not
# without_unsafe_list is a list containing possible without phrase might cause errors 
without_unsafe_list = ['without history of tuberous sclerosis']
without_unsafe_list = without_unsafe_list + ['without any evidence']
without_unsafe_list = without_unsafe_list + ['no history of glaucoma']
without_unsafe_list = without_unsafe_list + ['without scleral pigment']
without_unsafe_list = without_unsafe_list + ['without any evidence of']
without_unsafe_list = without_unsafe_list + ['without an intervening']
without_unsafe_list = without_unsafe_list + ['without ocular symptoms']
without_unsafe_list = without_unsafe_list + ['without the interposition of arterioles or capillaries']

# without_safe_list is a list containing without phrase , but this phrase did not cause error for 'DS' 
without_safe_list = ['without success.','without therapy.', 'without any care before.','without any problems.', 'without treatment', 'without cnvm', 'without known', 'without history of trauma', 'without symptom', 'without improvement']
without_safe_list = without_safe_list + ['without significant']
without_safe_list = without_safe_list + ['without pressure']
without_safe_list = without_safe_list + ['without correction']
without_safe_list = without_safe_list + ['without cell']
without_safe_list = without_safe_list + ['without pain']
without_safe_list = without_safe_list + ['without drusen']
without_safe_list = without_safe_list + ['without recovery']
without_safe_list = without_safe_list + ['without any recurrence']
without_safe_list = without_safe_list + ['without wide-field imaging']
without_safe_list = without_safe_list + ['without plaquinel']
without_safe_list = without_safe_list + ['without any late leakage']
without_safe_list = without_safe_list + ['without holes']
without_safe_list = without_safe_list + ['without scleral indentation']
without_safe_list = without_safe_list + ['without recurrence']
without_safe_list = without_safe_list + ['without flecks']
without_safe_list = without_safe_list + ['without direct']
without_safe_list = without_safe_list + ['without subretinal new']
without_safe_list = without_safe_list + ['without edema']
without_safe_list = without_safe_list + ['without syptoms']
without_safe_list = without_safe_list + ['without any retinal crystals']
without_safe_list = without_safe_list + ['without intratetinal edema']
without_safe_list = without_safe_list + ['without prp']
without_safe_list = without_safe_list + ['without any underlying']
without_safe_list = without_safe_list + ['without early hyperfluorescent']
without_safe_list = without_safe_list + ['without any glial tissue']
without_safe_list = without_safe_list + ['without a definite mass lesion']
without_safe_list = without_safe_list + ['without an enlarged blindspot']
without_safe_list = without_safe_list + ['without hemorrhage']
without_safe_list = without_safe_list + ['without intraocular lens']
without_safe_list = without_safe_list + ['without specific treatment']
without_safe_list = without_safe_list + ['without signs of cnvm']
without_safe_list = without_safe_list + ['without any ophthalmological']
without_safe_list = without_safe_list + ['without an apd']
without_safe_list = without_safe_list + ['without any hx/fhx']
without_safe_list = without_safe_list + ['without any signs']
without_safe_list = without_safe_list + ['without foveolar luster']
without_safe_list = without_safe_list + ['without vitreous seeding']
without_safe_list = without_safe_list + ['without any sr exudate']
without_safe_list = without_safe_list + ['without any shadowing']
without_cnt_copy = without_cnt.copy()

# the following code is to check whether safe word has been removed
for key, value in without_cnt.items():
    print_flag = True
    for safe_item in without_safe_list:
        if safe_item in key:
            del(without_cnt_copy[key])
            print_flag = False
    # if print_flag:
    #     print(key, value)
# print('the length of without dictionary after removing words', len(without_cnt_copy))


# print the unsafe list
unsafe_id_list = []
safe_id_list = []
print('the following ids might cause problems')
for single_dic in cleaned_csv_list:
    for unsafe_text in without_unsafe_list:
        if unsafe_text in single_dic['text'] and len(single_dic['DS']) > 1:
            unsafe_id_list.append(single_dic['id'])
print(unsafe_id_list)

print('the following ids are safe')
for single_dic in cleaned_csv_list:
    for safe_text in without_safe_list:
        if safe_text in single_dic['text'] and len(single_dic['DS']) > 1:
            safe_id_list.append(single_dic['id'])
print(safe_id_list)
