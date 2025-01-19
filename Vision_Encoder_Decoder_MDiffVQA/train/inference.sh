# conda activate hroc

python inference.py \
    --exp_name EfficientNet \
    --model_arch EffNetBert \
    --load_weights /home/maasala/RadiologyQA/EXPERIMENTS/EfficientNet/best_meteor_9_model.pt \
    # --metrics_on_train