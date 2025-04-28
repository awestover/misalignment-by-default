apt update; apt install vim -y;
python3 -m venv env;
. env/bin/activate;
pip install accelerate; 
pip install torch; 
pip install transformers; 
pip install datasets; 
pip install huggingface_hub; 
huggingface-cli login
