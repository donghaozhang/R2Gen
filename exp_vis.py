import json
import difflib
import pandas as pd
import os
from modules.metrics import compute_scores
from tqdm import tqdm
json_path = '/media/hdd/donghao/imcaption/R2Gen/results/iu_xray_debug_eval/epoch_1_results.json'
with open(json_path) as f:
  json_data = json.load(f)
# print(json_data.keys())
showid = 1
single_pred_report = json_data['pred_report'][showid]
single_gt_report = json_data['gt_report'][showid]
single_image_id = json_data['images_id'][showid]
# print('single_pred_report', single_pred_report)
# print('single_gt_report', single_gt_report)
# print('single_image_id', single_image_id)
# print('verify the length of predicted reports', len(json_data['pred_report']))
# print('verify the length of ground truth reports', len(json_data['gt_report']))
# print('verify the length of images', len(json_data['images_id']))

# seq = difflib.SequenceMatcher(None,value,input_DS)
# score = seq.ratio()*100
total_testing_num = len(json_data['images_id'])
maxscore = -1
pair_report_pred = 'init'
pair_report_gt = 'init'
pair_image_id = -1
report_pred_list = []
report_gt_list = []
diff_scores_list = []
for i in tqdm(range(total_testing_num)):
	single_pred_report = json_data['pred_report'][i]
	single_gt_report = json_data['gt_report'][i]
	single_image_id = json_data['images_id'][i]
	seq = difflib.SequenceMatcher(None,single_pred_report,single_gt_report)
	score = seq.ratio()*100
	if score > maxscore:
		maxscore = score
		pair_report_pred = single_pred_report
		pair_report_gt = single_gt_report
		pair_image_id = single_image_id
	report_pred_list.append(single_pred_report)
	report_gt_list.append(single_gt_report)
	diff_scores_list.append(score)
	eval_res = compute_scores(gts={1:[single_gt_report]}, res={1:[single_pred_report]})
print(type(single_gt_report))
# print('the closest report prediction is', pair_report_pred)
# print('the closest ground truth is', pair_report_gt)
# print('the closest image id is', pair_image_id)
# print('the cloest score is', score)
results_dict = {'Report Prediction': report_pred_list,
				'Report Ground Truth': report_gt_list,
				'Matching Scores': diff_scores_list}
df = pd.DataFrame(results_dict)
# df.to_csv("results/iu_xray_debug_eval/file_name.csv")
dirname = os.path.dirname(json_path)
json_basename = os.path.basename(json_path)
json_basename_no_ext = json_basename.split('.')[0]
csv_path = os.path.join(dirname, json_basename_no_ext+'.csv')
df.to_csv(csv_path)
