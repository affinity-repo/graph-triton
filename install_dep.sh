conda create -n graph-triton
conda run -n graph-triton conda install python=3.9.13 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda run -n graph-triton pip install expecttest hypothesis pytest triton absl-py transformers overrides loguru
export PYTHONPATH=/home/wenxh/graph-triton/