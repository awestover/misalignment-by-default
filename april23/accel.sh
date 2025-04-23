accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_sharding_strategy 1 \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_backward_prefetch BACKWARD_PRE \
  --fsdp_state_dict_type SHARDED_STATE_DICT \
  2-ultra-tune.py