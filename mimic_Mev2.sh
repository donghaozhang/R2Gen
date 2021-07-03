CUDA_VISIBLE_DEVICES=1 python main.py \
--model mev2 \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 16 \
--epochs 2 \
--num_layers 3 \
--num_heads 8 \
--save_dir results/mev2_mimic \
--step_size 1 \
--gamma 0.8 