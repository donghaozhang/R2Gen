CUDA_VISIBLE_DEVICES=1 python main.py \
--model r2genaugv3 \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--lr_ed 5e-5 \
--batch_size 40 \
--epochs 200 \
--save_dir results/iu_xray_augv3 \
--step_size 50 \
--gamma 0.1 \
--n_gpu 1 \
# --seed 9223