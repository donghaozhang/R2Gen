import csv
import os
import json
from random import random

def get_label():
    num = random()
    if num >= 0 and num < 0.7:
        label = 'train'
    elif num >= 0.7 and num < 0.9:
        label = 'val'
    elif num >= 0.9 and num <= 1:
        label = 'test'
    return label

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

print('the length of dictionary', len(cleaned_csv_list))
len_list = [1, 5, 9, 'x','y','z', 20, 25, {}]
print ("Number of items in the list = ", len(len_list))
len_list.append({})
print ("Number of items in the list = ", len(len_list))

# iterate through the list
# convert cleaned_report_csv into the required json
danli_dataset_list = {} 
train_list = []
val_list = []
test_list = []
for i in range(len(cleaned_csv_list)):
    cur_label = get_label()
    cur_dict = cleaned_csv_list[i]
    r2gen_format_dict = {}
    if cur_label == 'train':
        r2gen_format_dict['id'] = str(i)
        r2gen_format_dict['report'] = cur_dict['caption']
        r2gen_format_dict['image_path'] = cur_dict['dir']
        train_list.append(r2gen_format_dict)
    elif cur_label == 'val':
        r2gen_format_dict['id'] = str(i)
        r2gen_format_dict['report'] = cur_dict['caption']
        r2gen_format_dict['image_path'] = cur_dict['dir']
        val_list.append(r2gen_format_dict)
    elif cur_label == 'test':
        r2gen_format_dict['id'] = str(i)
        r2gen_format_dict['report'] = cur_dict['caption']
        r2gen_format_dict['image_path'] = cur_dict['dir']
        test_list.append(r2gen_format_dict)
    if len(r2gen_format_dict['image_path']) < 5:
        print('r2gen_format_dict[image_path]', r2gen_format_dict['image_path'])
danli_dataset_list['train'] = train_list
danli_dataset_list['val'] = val_list
danli_dataset_list['test'] = test_list
with open('data/annotation.json', 'w', encoding='utf-8') as f:
    json.dump(danli_dataset_list, f, ensure_ascii=False, indent=4)
# load the iuxray file to understand the json file structure
# iuxray_path = '/media/hdd/donghao/imcaption/R2Gen/data/iu_xray/annotation.json'
# ann = json.loads(open(iuxray_path, 'r').read())
# print(ann.keys())
# print('the length of training dataset', len(ann['train']))
# print('the length of validating dataset', len(ann['val']))
# print('the length of testing dataset', len(ann['test']))
cur_label1 = get_label()
cur_label2 = get_label()
cur_label3 = get_label()
print('cur_label1', cur_label1, 'cur_label2', cur_label2, 'cur_label3', cur_label3)
