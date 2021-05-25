import os
from os import listdir
from os.path import isfile, join
import cv2
folder_name = "/media/hdd/donghao/imcaption/R2Gen/data/danli_datav2/images"
save_folder_name = "/media/hdd/donghao/imcaption/R2Gen/data/clean_danli_datav2" 
mypath = folder_name
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(onlyfiles)
NoneType = type(None)
dim = (256, 256)
for i, cur_fname in enumerate(onlyfiles):
	impath = os.path.join(folder_name, cur_fname)
	# print(impath)
	img = cv2.imread(impath)
	# print(img.shape[2])
	# if isinstance(img, NoneType):
	# 	print(impath)
	# if len(img.shape) == 2:
	# 	print('bingo', len(img.shape))
	if img.shape[2] == 4:
		print('bingo')
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	im_name_wo_ext = os.path.splitext(cur_fname)[0]
	im_name = im_name_wo_ext + '.png'
	save_im_path = os.path.join(save_folder_name, im_name)
	cv2.imwrite(save_im_path, resized)
