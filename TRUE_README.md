conda create -n video_retrieval python=3.9 -y
conda activate video_retrieval
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install --no-build-isolation git+https://github.com/openai/CLIP.git


conda create -n lavis python=3.9 -y
conda activate lavis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


