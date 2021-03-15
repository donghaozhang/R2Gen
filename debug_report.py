import json
ann_path = '/media/hdd/donghao/imcaption/R2Gen/data/mimic_cxr/annotation.json'
report = json.loads(open(ann_path, 'r').read())
print(report.keys())
print(report['train'][0])
# self.examples = self.ann[self.split]
# for i in range(len(self.examples)):
#     self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
#     self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])