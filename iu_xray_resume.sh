python main.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 1 \
--save_dir results/iu_xray_debugv2 \
--step_size 50 \
--gamma 0.1 \
--n_gpu 1 \
--save_period 2 
# --resume results/iu_xray_debug/model_best.pth \
# --seed 9223