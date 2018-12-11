import numpy as np
from pandas import DataFrame
import gym
from gym.envs.registration import register, spec
import logging
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
)

ACTION_LEFT = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DEFAULT = ACTION_LEFT
ACTION_TEXT = {
    ACTION_LEFT: 'left',
    ACTION_DOWN: 'down',
    ACTION_RIGHT: 'right',
    ACTION_UP: 'up'
}

logger = logging.getLogger('log')
logger.setLevel(logging.WARNING)
np.random.seed(12321)

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class FrozenLake:
    def __init__(self, env):
        self.stateCnt      = env.observation_space.n
        self.actionCnt     = env.action_space.n # left:0; down:1; right:2; up:3
        self.LEARNING_RATE = 1
        self.GAMMA         = 0.5
        self.EPSILON       = 0.5
        self.Q             = self._initialiseModel()

    def _initialiseModel(self):
        return DataFrame(np.zeros((self.stateCnt, self.actionCnt)))

    def predict_value(self, status):
        valid_actions = [ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP]

        if status < 4:
            valid_actions.remove(ACTION_UP)
        if status % 4 == 0:
            valid_actions.remove(ACTION_LEFT)
        if status >= 12:
            valid_actions.remove(ACTION_DOWN)
        if (status + 1) % 4 == 0:
            valid_actions.remove(ACTION_RIGHT)

        return valid_actions

    def choose_action(self, status, choose_best = False):
        status_Q = self.Q.loc[status, :]
        valid_actions = self.predict_value(status)
        action = ACTION_DEFAULT

        first = (status_Q == 0).all()

        if_explore = False
        if choose_best:
            if_explore = False
        elif first:
            if_explore = True
        else:
            if_explore = np.random.uniform() < self.EPSILON

        if if_explore:
            # exploration
            action = np.random.choice(valid_actions)
        else:
            # exploitation
            max_Q = -1
            for a in valid_actions:
                if status_Q.loc[a] > max_Q:
                    action = a
                    max_Q = status_Q.loc[a]

        return action

    def update_value_Qlearning(self, status, action, reward, next_status):
        self.Q.loc[status, action] += self.LEARNING_RATE * (reward + self.GAMMA * self.Q.loc[next_status, :].max() - self.Q.loc[status, action])

    def update_value_SARSA(self, status, action, reward, next_status, next_action):
        self.Q.loc[status, action] += self.LEARNING_RATE * (reward + self.GAMMA * self.Q.loc[next_status, next_action]- self.Q.loc[status, action])

    def run_episode_qlearning(self):
        env.reset()
        status = 0
        episodeStepsCnt = 0
        r_total = 0

        while True:
            episodeStepsCnt += 1
            action = self.choose_action(status)

            #env.render()
            next_status, reward, done, info = env.step(action)

            # -1 reward when get in hole
            if done and reward == 0:
                reward = -1
            r_total += reward
            self.update_value_Qlearning(status, action, reward, next_status)
            status = next_status
            if done:
                break
        return r_total, episodeStepsCnt

    def run_episode_sarsa(self):
        env.reset()
        status = 0
        episodeStepsCnt = 0
        r_total = 0
        action = self.choose_action(status)
        while True:
            episodeStepsCnt += 1
            #env.render()
            next_status, reward, done, info = env.step(action)
            # -1 reward when get in hole
            if done and reward == 0:
                reward = -1
            r_total += reward
            next_action = self.choose_action(next_status)
            self.update_value_SARSA(status, action, reward, next_status, next_action)
            status = next_status
            action = next_action
            if done:
                break
        return r_total, episodeStepsCnt

    def run_evaluation_episode(self):
        reach_num = 0
        total_steps = 0

        for _ in range(10):
            is_record = True if _ == 9 else False
            actions = []

            env.reset()
            status = 0
            steps = 0
            reach_goal = False

            while True:
                action = self.choose_action(status, choose_best=True)
                if is_record:
                    actions.append(ACTION_TEXT[action])
                next_status, reward, done, _ = env.step(action)

                if status == next_status:
                    continue

                steps += 1

                status = next_status

                if done:
                    reach_goal = (reward != 0)
                    break

            if reach_goal:
                reach_num += 1
                total_steps += steps

        average_steps = (0 if reach_num == 0 else (total_steps / reach_num))
        print("{}/10 reached, average steps {}".format(reach_num, average_steps))
        print("Best Path: {}".format(actions))

if __name__ == '__main__':
    env                      = gym.make('FrozenLakeNotSlippery-v0')
    frozenlake               = FrozenLake(env)
    r_total_progress         = []
    episodeStepsCnt_progress = []
    q_Sinit_progress         = []
    nbOfTrainingEpisodes     = 10240
    for i in range(nbOfTrainingEpisodes):
        #  decrease EPSILON
        if i%1024 == 0:
            frozenlake.EPSILON -= 0.05
            print(frozenlake.EPSILON)
        r_total, episodeStepsCnt = frozenlake.run_episode_qlearning()
        #r_total, episodeStepsCnt = frozenlake.run_episode_sarsa()
        r_total_progress.append(r_total)
        episodeStepsCnt_progress.append(episodeStepsCnt)
        q_Sinit_progress.append(frozenlake.Q.loc[0].values.tolist())
        if (i+1) % 256 == 0:
            frozenlake.run_evaluation_episode()
        #if (i+1) % 1024 == 0:
        #    print(frozenlake.Q)
        
    ### --- Plots --- ###
    # 1) plot world.q_Sinit_progress
    fig1 = plt.figure(1)
    plt.ion()
    plt.plot([q[0] for q in q_Sinit_progress], label='left',  color = 'r')
    plt.plot([q[1] for q in q_Sinit_progress], label='down',  color = 'g')
    plt.plot([q[2] for q in q_Sinit_progress], label='right', color = 'b')
    plt.plot([q[3] for q in q_Sinit_progress], label='up',    color = 'y')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop = fontP, loc=1)
    plt.pause(0.001)

    # 2) plot the evolution of the number of steps per successful episode throughout training. A successful episode is an episode where the agent reached the goal (i.e. not any terminal state)
    fig2 = plt.figure(2)
    plt1 = plt.subplot(1,2,1)
    plt1.set_title("Number of steps per successful episode")
    plt.ion()
    plt.plot(episodeStepsCnt_progress)
    plt.pause(0.0001)
    # 3) plot the evolution of the total collected rewards per episode throughout training. you can use the running_mean function to smooth the plot
    plt2 = plt.subplot(1,2,2)
    plt2.set_title("Rewards collected per episode")
    plt.ion()
    r_total_progress = running_mean(r_total_progress)
    plt.plot(r_total_progress)
    plt.pause(10)
    ### --- ///// --- ###