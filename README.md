# Reinforcement Learning with Policy Gradient
### Uses a 3 layer neural network as the policy network

The idea is to create a deep policy network that is intelligent enough to generalize to most games in OpenAI's Gym.

1) To run this code first install OpenAI's Gym: https://github.com/openai/gym

2) Download this repo and run `python run_carpole.py` to run the agent (or any other game in this repo, like `python run_lunarlander.py`) and see it improve over time.

3) To run a Box2D game like LunarLander you have to install the Box2D Physics engine: `pip install -e '.[box2d]'`

## Lunar Lander
### Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible and fuel is infinite.

Initially, the agent is as good as randomly picking the next action:
![pg-ll-initial](https://user-images.githubusercontent.com/1076706/33915897-e3e8f95c-df5a-11e7-9b73-d4287c5e99d9.gif)

After several hundred episodes, the agent starts learning how to fly and hover around:
![pg-ll-learning_to_fly](https://user-images.githubusercontent.com/1076706/33915899-e74cb58e-df5a-11e7-8549-236dcf379212.gif)

Finally after about 3K episodes the agent can land pretty well:
![pg-ll-landings](https://user-images.githubusercontent.com/1076706/33915900-ea25fd06-df5a-11e7-9c7a-71dafc04a770.gif)

## Cartpole
### A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

Initially, the agent is quite dumb, but it's exploring the state/action/reward space:
![Untrained](https://user-images.githubusercontent.com/1076706/33820098-7e43df3a-de02-11e7-81fe-970f6df33e1c.gif)

As more episodes go by, it starts to get better by learning from experience (using reward guided loss):
![pg-cartpole-2](https://user-images.githubusercontent.com/1076706/33820104-83fa04e0-de02-11e7-9dca-34f7a3f45226.gif)

Eventually, the agent masters the game (trained on my Macbook Pro for ~10 minutes):
![pg-cartpole-trained](https://user-images.githubusercontent.com/1076706/33820246-4bd347e2-de03-11e7-825d-58c212c346c6.gif)

After 297 episodes the agent scored 617,332!

![record](https://user-images.githubusercontent.com/1076706/33820269-67d42344-de03-11e7-903e-bbf9b8e0ab9b.png)
