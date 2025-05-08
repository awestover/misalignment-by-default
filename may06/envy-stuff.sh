# Set Hugging Face cache directory
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics

# Optional: set PyTorch Hub cache if needed
export TORCH_HOME=/workspace/torch_cache

# Alias for convenience
alias py=python

# Confirmation
echo "Environment variables and alias for Hugging Face, PyTorch, and 'py' set."
