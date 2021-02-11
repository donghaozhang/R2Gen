CUDA_VISIBLE_DEVICES=0 python main.py \
--model r2genaugv3abrm \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 60 \
--threshold 3 \
--batch_size 40 \
--epochs 30 \
--save_dir results/mimic_augv3abrm \
--step_size 50 \
--gamma 0.1 \
--n_gpu 1 \
# --seed 9223