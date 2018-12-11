# FrozenLake_OpenAI_Gym
使用OpenAI Gym实现Frozen Lake环境的修改版本

### 代码实现
+ predict_value(s): returns a vector with the value of each action in state s.
+ update_value_Q(s, a, r, s_next): updates the current estimate of the value of the
state-action pair <s,a> using Q-learning.
+ update_value_S(s, a, r, s_next): updates the current estimate of the value of the
state-action pair <s,a> using either Sarsa.
+ choose_action(s): returns the action to execute in state s, implementing an ε-greedy
policy.
+ run_episode_qlearning(): runs an episode, learning with the Q-learning algorithm.
+ run_episode_sarsa(): runs an episode, learning with the Sarsa algorithm.
+ run_evaluation_episode(): runs an episode executing the currently optimal policy.

### 版本条件
* VS code
* Python 3.7.1
* gym 0.10.9
* matplotlib 3.0.2
* numpy 1.15.4
* pandas 0.23.4

##### [报告下载](https://report-1257390182.cos.ap-chengdu.myqcloud.com/frozenlake%E6%8A%A5%E5%91%8A.docx)
