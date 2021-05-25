import simple_icd_10 as icd
import csv
import os
import json
from random import random
import difflib


def cal_diff_score(input_DS, icd_dict):
	minscore = -1
	for key, value in eye_icd_dict.items():
		seq = difflib.SequenceMatcher(None,value,input_DS)
		score = seq.ratio()*100
		if score > minscore:
			minscore = score
			close_icd = value
	print(close_icd)
	return close_icd

# https://pypi.org/project/simple-icd-10/
eye_icd_dict = {}
output = icd.get_description("H25")
print(output)
for i in range(0, 6):
	# print(i)
	for j in range(10):
		# print(j)
		curstr = 'H' + str(i) + str(j) 
		# print(curstr)
		valid_flag = icd.is_valid_code(curstr)
		if valid_flag:
			eye_desc = icd.get_description(curstr)
			# print(eye_code)
			eye_icd_dict[curstr] = eye_desc

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
            close_icd = cal_diff_score(single_dic['DS'] , eye_icd_dict)
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
