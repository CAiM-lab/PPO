# (c) 2019-​2020,   Emanuel Joos  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
# Based on https://github.com/higgsfield/RL-Adventure-2
from network import ActorCritic
import torch
import numpy as np
from multiprocessing_env import SubprocVecEnv
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pickle
import torch.nn as nn
import random
import time
import pdb
import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
logger.setLevel(logging.INFO)


class PPO(object):
    """Main PPO class"""
    def __init__(self, args):
        """"Constructor which allows the PPO class to initialize the attributes of the class"""
        self.args = args
        self.random_seed()
        # Check if GPU is available via CUDA driver
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # Initialize the actor critic class
        self.actor_critic = ActorCritic(self.args.nb_states,
                                        self.args.nb_actions,
                                        self.args.hidden_layer_size).to(self.device)
        # Define the optimizer used for the optimization of the surrogate loss
        self.optimizer = self.args.optimizer(self.actor_critic.parameters(), self.args.lr)

        # For training multiple instances of the env are needed (Shoulder model)
        self.envs = [self.make_env() for i in range(self.args.num_envs)]
        self.envs = SubprocVecEnv(self.envs)
        # To validate the intermediate learning process one test env is needed
        self.env_test = self.args.env
        self.env_test.seed(self.args.seed)
        self.env_test.set_scaling(self.args.output_scaling)

        #  Lists for Tensorboard to visualize learning process during learning
        self.test_rewards = []
        self.loss = []
        self.lr = []
        self.actor_grad_weight = []
        self.action_bang_bang = []

        self.lr.append(self.args.lr)

        # Dump bin files
        if self.args.play is False:
            self.output_path = "trained_models" + '/PPO_{}'.format(datetime.now().strftime('%Y%b%d_%H%M%S')) + "/"
            os.mkdir(self.output_path)
            self.writer = SummaryWriter(self.output_path)

        #self.delta = (self.args.lr-self.args.lr_end)/1e6

    def train(self):
        """Main training function"""
        frame_idx = 0
        state = self.envs.reset()
        mean_100_reward = -np.inf
        self.info()

        while frame_idx < self.args.max_frames:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = self.args.entropy

            for _ in range(self.args.nb_steps):
                state = torch.FloatTensor(state).to(self.device)
                dist, value = self.actor_critic(state)
                action = dist.sample()
                # Make sure action is loaded to CPU (not GPU)
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))

                states.append(state)
                actions.append(action)
                state = next_state
                frame_idx += 1
                #self.scheduler()

                # Evaluate training process and write data to tensorboard
                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test_env(self.args.vis) for _ in range(10)])
                    self.test_rewards.append(test_reward)

                    if self.args.play is False:
                        print("Mean reward: ", np.round(np.mean(self.test_rewards[-101:-1]), 0))
                        if mean_100_reward < np.round(np.mean(self.test_rewards[-101:-1]), 0):
                            mean_100_reward = np.round(np.mean(self.test_rewards[-101:-1]), 0)
                            self.save_network(mean_100_reward)
                        if len(self.test_rewards) >= 10:
                            self.writer.add_scalar('data/reward', np.mean(self.test_rewards[-11:-1]),
                                                   frame_idx*self.args.num_envs)
                            self.writer.add_scalar('data/ppo_loss', np.mean(self.loss[-11:-1]),
                                                   frame_idx*self.args.num_envs)
                            self.writer.add_scalar('data/nb_actions_outside_range', np.mean(self.action_bang_bang[-11:-1]),
                                                   frame_idx*self.args.num_envs)

                    # if test_reward > threshold_reward: early_stop = True

            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.actor_critic(next_state)
            returns = self.calc_gae(next_value, rewards, masks, values, self.args.gamma, self.args.tau)

            # detach() to take it away from the graph i.e. this operations are ignored for gradient calculations
            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values
            self.ppo_update(self.args.ppo_epochs, self.args.mini_batch_size, states, actions, log_probs, returns,
                            advantage, self.args.clip)

    def make_env(self):
        # Private trunk function for calling the SubprocVecEnv class
        def _trunk():
            env = self.args.env # in this simple case the class TestEnv() is called (see openAI for more envs)
            env.seed(self.args.seed)
            env.set_scaling(self.args.output_scaling)
            return env
        return _trunk

    def test_env(self, vis=False):
        state = self.env_test.reset()
        if vis:
            self.env_test.render()
        done = False
        total_reward = 0
        action_bang_bang = 0
        step = 0
        while not done:
            step+=1
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist, _ = self.actor_critic(state)
            action = dist.sample().cpu().numpy()[0]
            force = action * self.args.output_scaling
            next_state, reward, done, _ = self.env_test.step(action)
            if force > 0.5 or force < -0.5:
                action_bang_bang += 1
            state = next_state
            if vis:
                self.env_test.render()
            total_reward += reward
        self.action_bang_bang.append(action_bang_bang/step)
        return total_reward
    # Plain functions except that one can call them from an instance or the class
    @staticmethod
    def calc_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    @staticmethod
    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                           rand_ids, :]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size,
                                                                                  states,
                                                                                  actions,
                                                                                  log_probs,
                                                                                  returns,
                                                                                  advantages):
                dist, value = self.actor_critic(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                self.loss.append(loss.item())
                # Important step:
                self.optimizer.zero_grad()
                #pdb.set_trace()
                loss.backward()
                if self.args.grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.grad_norm)
                self.optimizer.step()

    def save_network(self, reward):
        network_path = self.output_path + "/network" + str(reward)
        pickle.dump(self.actor_critic.state_dict(), open(network_path, "wb"))

    def load_network(self, path):
        network_new = pickle.load(open(path, "rb"))
        self.actor_critic.load_state_dict(network_new)

    def random_seed(self):
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

    def scheduler(self):
        for g in self.optimizer.param_groups:
            lr = g["lr"]
            if self.args.lr_end > lr:
                lr = self.args.lr_end
            else:
                lr -= self.delta
            self.lr.append(lr)
            g["lr"] = lr
            
    def info(self):
        fhandler = logging.FileHandler(filename=self.output_path + '/mylog.log', mode='a')
        logger.addHandler(fhandler)
        logger.info("--- INFO ---")
        logger.info("args: {}".format(self.args))

