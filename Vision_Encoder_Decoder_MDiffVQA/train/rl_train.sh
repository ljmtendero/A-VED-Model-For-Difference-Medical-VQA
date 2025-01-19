python mytrain_rl.py \
    --exp_name exp21Penc2dconLRplateaugeneratewithdiffimagebatchsubido_RLDiferenteponderacion \
    --model_arch SwinBERT9k \
    --load_weights exp10Penc2dconLRplateaugeneratewithdiffimagebatchsubido/best_cider_7_model.pt\
    --scores_weights 0.7,0.2,0.1 \
    --scores BertScorer,F1RadGraph \
    --scores_args {},{\"reward_level\":\"partial\"} \
    --use_nll True \
    --top_k 0