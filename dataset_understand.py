import csv
import os
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
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
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
    print(f'Processed {line_count} lines.')
# print('the title is', column_title)
# print(img_folder_path, os.path.basename(single_dic['dir']))
# print(cleaned_csv_dict[1])
# cur_dict = single_dic.copy()
print('cur_dict[dir]', single_dic['dir'])
print(single_dic)
# print('row[0]', row[0], 'row[1]', row[1], 'row[2]', row[2], 'row[3]', row[3], 'row[4]', row[4], 'row[5]', row[5])

# convert cleaned_report_csv into the required json
# print(cleaned_csv_dict)
print('the length of dictionary', len(cleaned_csv_list))
len_list = [1, 5, 9, 'x','y','z', 20, 25, {}]
print ("Number of items in the list = ", len(len_list))
len_list.append({})
print ("Number of items in the list = ", len(len_list))