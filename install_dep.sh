conda create -n graph-triton
conda run -n graph-triton conda install python=3.9.13 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda run -n graph-triton pip install expecttest hypothesis pytest triton absl-py transformers overrides loguru
export PYTHONPATH=/home/wenxh/graph-triton/