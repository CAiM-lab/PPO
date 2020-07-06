# PPO : Proximal Policy Optimization 
## Introduction
This is our PPO implementation used for reinforcement learning of a controller in:\
> Emanuel Joos, Fabien Pean, Orcun Goksel: 
"Reinforcement Learning of Musculoskeletal Control from Functional Simulations", 2020.

If you use this code, please cite the work above.\
Installation instructions and a simple demo example are given below.
## Set up python environment
1. Clone OpenAI gym and OpenAI baselines from:\
https://github.com/openai/baselines \
https://github.com/openai/gym/ 
2. Copy the test_envs.py to OpenAI/gym_shoulder/gym_OneMuscle/envs/ (Create the directory first)
3. Copy the init.py to the same directory
4. Copy the setup.py to OpenAI/gym_shoulder/
5. Download and install conda
6. Set up the virtual environment
```console
conda create -n PPO python=3.5
conda activate PPO
pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl --user
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl --user
pip install seaborn psutil matplotlib pandas tensorboard tensorboardX
pip install --user ipykernel
python -m ipykernel install --user --name=PPO
cd OpenAI/baselines
pip install -e .
cd OpenAI/gym
pip install -e .
```
## Train a simple control example
```console
source activate PPO
cd PPODummyExample
python main.py --vis True
```
Training is very slow if vis is set true.
For all availabe arguments please have a look at PPODummyExample/main.py

## Run the pretrained example for control
```console
source activate PPO
cd PPODummyExample
python main.py --play True
```
![Simple control example](/videos/Dummy.gif)

<img src="/videos/CAiM_logo.png" width="300">
