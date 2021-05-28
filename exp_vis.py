import json
import difflib
import pandas as pd
import os
from modules.metrics import compute_scores
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu

json_path = '/media/hdd/donghao/imcaption/R2Gen/results/iu_xray_debug_eval/epoch_1_results.json'
with open(json_path) as f:
  json_data = json.load(f)
showid = 1
single_pred_report = json_data['pred_report'][showid]
single_gt_report = json_data['gt_report'][showid]
single_image_id = json_data['images_id'][showid]
# print('verify the length of predicted reports', len(json_data['pred_report']))
# print('verify the length of ground truth reports', len(json_data['gt_report']))
# print('verify the length of images', len(json_data['images_id']))

total_testing_num = len(json_data['images_id'])
# total_testing_num = 1
maxscore = -1
pair_report_pred = 'init'
pair_report_gt = 'init'
pair_image_id = -1
report_pred_list = []
report_gt_list = []
diff_scores_list = []
BLEU_1_score_list = []
BLEU_2_score_list = []
BLEU_3_score_list = []
BLEU_4_score_list = []
image_id_list = []
metrics = Bleu(4)
method = ['BLEU_1','BLEU_2','BLEU_3','BLEU_4']
compute_all_metric_flag = False
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
	if compute_all_metric_flag:
		eval_res = compute_scores(gts={1:[single_gt_report]}, res={1:[single_pred_report]})
	score, _ = metrics.compute_score(gts={1:[single_gt_report]}, res={1:[single_pred_report]}, verbose=0)
	# test_met = dict(zip(method,score))
	# print(score)
	BLEU_1_score_list.append(score[0])
	BLEU_2_score_list.append(score[1])
	BLEU_3_score_list.append(score[2])
	BLEU_4_score_list.append(score[3])
	# image_id_list.append()
	image_id_list.append(single_image_id)
results_dict = {'Image ID': image_id_list,  
				'Report Prediction': report_pred_list,
				'Report Ground Truth': report_gt_list,
				'Matching Scores': diff_scores_list,
				'BLEU_1': BLEU_1_score_list,
				'BLEU_2': BLEU_2_score_list,
				'BLEU_3': BLEU_3_score_list,
				'BLEU_4': BLEU_4_score_list}
df = pd.DataFrame(results_dict)
dirname = os.path.dirname(json_path)
json_basename = os.path.basename(json_path)
json_basename_no_ext = json_basename.split('.')[0]
csv_path = os.path.join(dirname, json_basename_no_ext+'.csv')
df.to_csv(csv_path)