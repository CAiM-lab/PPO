# (c) 2019-​2020,   Emanuel Joos  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
from ppo import PPO
import torch.optim as optim
import torch.nn as nn
import torch
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../OpenAI/'))
from OpenAI.gym_shoulder.gym_OneMuscle.envs import test_envs
import argparse


if __name__ == '__main__':
    """The main function is a starting point for the PPO learning or playing schema.
    Description: 
        PPO: Proximal Policy Optimization (more information on the algorithm can be found here:
         https://arxiv.org/abs/1707.06347)
        Environment: Simple test environment of a single joint robot arm which is torque controlled in order to achieve
         any desired angle. 
    Arguments: 
        All the arguments are passed via an argument parser. For description type python main.py --help
    """
    parser = argparse.ArgumentParser(description='PyTorch implementation of ppo for muscular skeleton model')
    parser.add_argument('--nb_states', default=4, type=int, help='How many states are in the env')
    parser.add_argument('--nb_actions', default=1, type=int, help='How many actions are in the env')
    parser.add_argument('--hidden_layer_size', default=256, type=int, help='Number of neurons in hidden layer')
    parser.add_argument('--optimizer', default=optim.Adam, type=str, help='Torch optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lr_end', default=0.00007, type=float, help='Final learning rate if learning rate decay')
    parser.add_argument('--num_envs', default=4, type=int, help='Number of envs in Parallel')
    parser.add_argument('--env', default=test_envs.TestEnv(), type=str, help='Environement')
    parser.add_argument('--max_frames', default=4000000, type=int, help='Max frames per env')
    parser.add_argument('--nb_steps', default=500, type=int,
                        help='After how many steps per environment is an update done')
    parser.add_argument('--ppo_epochs', default=4, type=int, help='Number of epochs')
    parser.add_argument('--mini_batch_size', default=250, type=int)
    parser.add_argument('--entropy', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.95, type=float)
    parser.add_argument('--clip', default=0.2, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--grad_norm', default=0.01, type=float)
    parser.add_argument('--play', default=False, type=bool)
    parser.add_argument('--load_path', default="trained_models/dummyNetwork/network-36.0", type=str)
    parser.add_argument('--output_scaling', default=1, type=float)
    parser.add_argument('--vis', default=False, type=bool, help="For visualizing the test env during training")

    args = parser.parse_args()
    # Arguments are passed to the PPO algorithm
    ppo = PPO(args)
    # If play is true, an already trained network is loaded and feed forward control is applied.
    if args.play:
        # Load weights into actor critic network
        ppo.load_network(args.load_path)
        state = ppo.env_test.reset()
        for t in range(750):
            ppo.env_test.render()
            state = torch.FloatTensor(state).unsqueeze(0).to(ppo.device)
            dist, _ = ppo.actor_critic(state)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = ppo.env_test.step(action)
            state = next_state
            if done:
                state = ppo.env_test.reset()
        ppo.env_test.close()
    # The main train function from the PPO class is called
    else:
        ppo.train()
