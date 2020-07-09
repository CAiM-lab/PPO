# (c) 2019-​2020,   Emanuel Joos  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel
"""
Simple robot arm example

"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time

class TestEnv(gym.Env):
    """
    Description:
        Control of a single joint robot arm.

    Observations:
        Type: Box(2)
        Num	Observation                 Min         Max
        0	Angle                      -pi/2         pi/2
        1	Angular velocity           -Inf         Inf

    Actions:
        Type: Continuous
        Action                  Min         Max
        Force                   0N          15N

    Rewards:
         Reward is -1 for every step taken
         Reward +1 for every step at the goal position

    Starting State:
        Random between -pi/2 and pi/2

    Episode Termination:
        Angle under -pi/2-0.1 and over pi/2+0.1
        Arm is at goal position over some time --> specify later
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self):
        self.gravity = 9.8
        self.massarm = 1.0
        self.legth = 1.0
        self.factor = 3.0/(2.0*self.massarm*self.legth)
        self.tau = 0.02  # seconds between state updates
        self.goal_counter = 0
        self.output_scaling = 1


        # Angle at which to fail the episode
        self.phi_max = math.pi/2 + 0.5
        self.phi_min = -math.pi/2 - 0.5

        #Action Space:
        self.max_force = 0.5
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force,
                                       shape=(1,), dtype=np.float32)

        #Observation Space:
        high = np.array([np.inf]*4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()

        #goal position:
        self.goal = 0.0
        self.phi_dot_goal = 0.0
        self.mean_100ep_reward = -100000.0
        self.change_goal_counter = 0
        self.reward_per_episode = 0
        self.episode_counter = 0


        self.viewer = None
        self.state = None
        self.step_counter = 0
        self.global_counter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state
        phi, phi_relative, phi_dot, force_state = state
        phi, phi_relative_old, phi_dot, force_state = state
        force = action*self.output_scaling
        reward = 0
        if force > 0.5 or force < -0.5:
            reward -= 1
        try:
            force = np.clip(force, -0.5, 0.5)[0]
        except:
            force = np.clip(force, -0.5, 0.5)

        F_applied = force + force_state

        phi_dot = phi_dot + self.tau*(F_applied / (self.massarm * self.legth**2)-self.gravity/self.legth*math.cos(phi))
        phi = phi + self.tau*phi_dot
        phi_relative = phi - self.goal

        self.state = (phi, phi_relative, phi_dot, F_applied)
        self.step_counter += 1
        self.global_counter +=1

        if phi >= self.phi_max or phi <= self.phi_min:
            reward -= 100.0
            done = True
            return np.array(self.state), reward, done, {}

        reward -= np.abs(phi_relative) 

        done = self.step_counter > 149
        done = bool(done)

        return np.array(self.state), reward, done, {}

    def reset(self):

        self.goal = self.np_random.uniform(low=-math.pi/2.0,
                                             high=0.0,
                                             size=(1,))[0]

        self.episode_counter+=1

        self.step_counter = 0
        force_state = 0
        phi = -math.pi/2

        phi_dot = 0
        phi_relative = phi - self.goal
        self.state = (phi, phi_relative, phi_dot, force_state)
        self.goal_counter = 0
        return np.array(self.state)

    def change_goal(self):
        print("Changed Goal : {} -> +{} -{}".format(
                    self.change_goal_counter, 0, self.change_goal_counter+1))
        self.change_goal_counter+=1

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_with = self.legth*3
        scale = screen_width/world_with
        arm_width = 20
        arm_legth = scale*self.legth
        radius_joint = 5

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #robot arm:
            l,r,t,b = -arm_width/2, arm_width/2, arm_width/2, -arm_legth+arm_width/2
            arm = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            arm.set_color(0.0, 0.0, 0.0)
            self.armtrans = rendering.Transform(translation=(screen_width/2.0, screen_width/2.0))
            arm.add_attr(self.armtrans)
            self.viewer.add_geom(arm)

            #joint
            self.joint = rendering.make_circle(radius_joint)
            self.jointtrans = rendering.Transform()
            self.joint.add_attr(self.jointtrans)
            self.joint.add_attr(self.armtrans)
            self.joint.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.joint)
            #goal
            self.end = rendering.make_circle(radius_joint*2)
            self.endtrans = rendering.Transform(translation=(screen_width/2.0+(arm_legth-arm_width)*math.cos(self.goal),
                                                             screen_width/2.0+(arm_legth-arm_width)*math.sin(self.goal)))
            self.end.add_attr(self.endtrans)
            self.end.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(self.end)

        if self.state is None: return None

        x = self.state

        self.armtrans.set_rotation(x[0]+math.pi/2.0)
        self.endtrans.set_translation(screen_width/2.0+(arm_legth-arm_width)*math.cos(self.goal),
                                      screen_width / 2.0 + (arm_legth - arm_width) * math.sin(self.goal))
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def set_scaling(self, output_scaling):
        self.output_scaling = output_scaling

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
