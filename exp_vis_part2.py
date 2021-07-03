import json
import difflib
import pandas as pd
import os
from modules.metrics import compute_scores
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
import matplotlib.pyplot as plt

# ban
json_path = '/media/hdd/donghao/imcaption/R2Gen/results/ban_comp/epoch_32_results.json'
with open(json_path) as f:
  json_data = json.load(f)
# print(json_data.keys())
# print(json_data['images_id'])
print('the length of images_id of ban', len(json_data['images_id']))


# xlinear
# json_path_xlinear = '/media/hdd/donghao/imcaption/R2Gen/results/xlinear_comp/epoch_44_results.json'
# with open(json_path_xlinear) as fxlinear:
#   json_data_xlinear = json.load(fxlinear)
# # print(json_data.keys())
# # print(json_data['images_id'])
# print('the length of images_id of xlinear', len(json_data_xlinear['images_id']))

# # iuxray dataset
# json_path_iuxray = '/media/hdd/donghao/imcaption/R2Gen/data/iu_xray/annotation.json'
# with open(json_path_iuxray) as f:
#   json_path_iuxray = json.load(f)
#   print(len(json_path_iuxray['test']))