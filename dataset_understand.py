cleaned_report_path = '/media/hdd/data/imcaption/danli_datav2/cleaned.csv'
import csv

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
            line_count += 1
            single_dic['id'] = row[0]
            
    print(f'Processed {line_count} lines.')
print('the title is', column_title)
print('row[0]', row[0], 'row[1]', row[1], 'row[2]', row[2], 'row[3]', row[3], 'row[4]', row[4], 'row[5]', row[5])
