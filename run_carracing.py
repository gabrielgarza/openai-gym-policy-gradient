import gym
from policy_gradient_layers import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CarRacing-v0')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


RENDER_ENV = False
EPISODES = 500
rewards = []
RENDER_REWARD_MIN = 150


if __name__ == "__main__":

    PG = PolicyGradient(
        n_x = 2304,
        n_y = env.action_space.shape[0],
        learning_rate=0.01,
        reward_decay=0.95
    )

    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0
        steps_taken_in_episode = 0


        while True:
            if RENDER_ENV: env.render()

            # 0. Preprocess observation state pixels
            # Convert color pixel observation into gray scale from (96,96,3) to (96,96)
            observation = np.dot(observation[...,:3], [0.299, 0.587, 0.114])
            # Downsample by factor of 2 - shape from (96,96) to (48,48)
            observation = observation[::2,::2]
            # Unroll the matrix into a vector with shape (2304,)
            observation = observation.astype(np.float).ravel()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            action_array = np.zeros(env.action_space.shape[0])
            action_array[action] = 1
            # print("action_array", action_array)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action_array)

            # print("reward", reward)


            # 4. Store transition for training
            PG.store_transition(observation, action, reward)

            if steps_taken_in_episode > 500: done = True

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)

                # 5. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                # Render env if we get to rewards minimum
                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True


                break

            # Save new observation
            observation = observation_

            # increase steps taken
            steps_taken_in_episode += 1
