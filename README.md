# RLmuscle

## Set up python environment
1. Clone OpenAI gym and openAI baselines
2. Copy the test_envs.py to OpenAI/gym_shoulder/gym_OneMuscle/envs/ (Create the directory first)
3. Copy the __init__.py to the same directory
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
