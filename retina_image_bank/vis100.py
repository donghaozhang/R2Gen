import os
import json
import random
import shutil
import csv

random.seed(10)
json_path = '/media/hdd/donghao/imcaption/R2Gen/data/danli_datav2/annotationv2_debug.json'
with open(json_path) as f:
  json_data = json.load(f)
print(json_data.keys())
# dataset_split = 'train'
dataset_split = 'test'
train_list = json_data[dataset_split]
train_list_random = random.sample(train_list, len(train_list))
train_100 = []
# shutil.copyfile('debug.py', 'debug_copy.py')
# train_im_folder_path = '/media/hdd/donghao/imcaption/R2Gen/retina_image_bank/mimcap_model/data/sample/train/image/'
test_im_folder_path = '/media/hdd/donghao/imcaption/R2Gen/retina_image_bank/mimcap_model/data/sample/test/image/'
final_impath = '/media/hdd/donghao/imcaption/R2Gen/data/clean_danli_datav2/'
with open('test_sample.csv', 'w', newline='') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',')
	spamwriter.writerow(['Image', 'Ground Truth', 'Prediction'])
	for i in range(0,100,1):
		source_impath = train_list_random[i]['image_path']
		report = train_list_random[i]['report']
		filename = os.path.splitext(os.path.basename(source_impath))[0]
		copy_source_path = final_impath  + filename + '.png'
		spamwriter.writerow([filename + '.png', report])
	# copy_target_path = train_im_folder_path + filename + '.png'
	# copy_target_path = test_im_folder_path + filename + '.png' 
	# shutil.copyfile(copy_source_path, copy_target_path)