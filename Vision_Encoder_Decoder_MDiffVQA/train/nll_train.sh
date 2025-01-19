# conda activate hroc

python mytrain_nll.py \
    --exp_name SwinBERT_Finetuned_Unfreezed3 \
    --model_arch SwinBERTFinetuned \
    --load_weights SwinBERTFinetuned4/best_bleu1_1_model.pt \
    --hnm True
    # --metrics_on_train