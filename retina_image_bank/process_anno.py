import json
import os
json_path = '/media/hdd/donghao/imcaption/R2Gen/data/danli_datav2/annotationv2_debug.json'
# json_path = 
with open(json_path) as f:
  json_data = json.load(f)
test_retina_189 = json_data['test']
# print(test_retina_189)
cleaned_imfolder_path = []
im_dir = '/media/hdd/donghao/imcaption/R2Gen/data/clean_danli_datav2/'
for pair in test_retina_189:
	# cur = pair
	ground_truth_report = pair['report']
	impath = pair['image_path']
	real_impath = os.path.join(im_dir, os.path.basename(impath))
print(real_impath)
# print(cur)