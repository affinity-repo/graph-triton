FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update -y 
RUN apt install zsh htop tmux curl wget build-essential clang cmake llvm lld ninja-build git gdb clangd -y
RUN curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh \
	&& bash Mambaforge-$(uname)-$(uname -m).sh -b -p /opt/miniforge/ \
	&& rm Mambaforge-$(uname)-$(uname -m).sh \
	&& /opt/miniforge/bin/conda init bash \
	&& /opt/miniforge/bin/conda config --set auto_activate_base true \
	&& /opt/miniforge/bin/conda config --set changeps1 false
RUN /opt/miniforge/bin/conda run -n base conda install python==3.10.10 numpy pybind11
RUN /opt/miniforge/bin/conda run -n base conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN /opt/miniforge/bin/conda run -n base pip install cupy-cuda12x jinja2 jedi_language_server loguru jsonpickle pytest autopep8
WORKDIR /root/
COPY . /root/welder
