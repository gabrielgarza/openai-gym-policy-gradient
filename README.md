# Reinforcement Learning with Policy Gradient

The idea is to create a deep policy network that is intelligent enough to generalize to most games in OpenAI's Gym.

1) To run this code first install OpenAI's Gym: https://github.com/openai/gym

2) Download this repo and run `python run_carpole.py` to run the agent (or any other game in this repo, like `python run_lunarlander.py`) and see it improve over time.

3) To run a Box2D game like LunarLander you have to install the Box2D Physics engine: `pip install -e '.[box2d]'`


## Cartpole

Initially, the agent is quite dumb, but it's exploring the state/action/reward space:
![Untrained](https://user-images.githubusercontent.com/1076706/33820098-7e43df3a-de02-11e7-81fe-970f6df33e1c.gif)

As more episodes go by, it starts to get better by learning from experience and doing less exploration(getting more greedy):
![pg-cartpole-2](https://user-images.githubusercontent.com/1076706/33820104-83fa04e0-de02-11e7-9dca-34f7a3f45226.gif)

Eventually, the agent masters the game (trained on my Macbook Pro for ~10 minutes):
![pg-cartpole-trained](https://user-images.githubusercontent.com/1076706/33820246-4bd347e2-de03-11e7-825d-58c212c346c6.gif)

After 297 episodes the agent scored 617,332!

![record](https://user-images.githubusercontent.com/1076706/33820269-67d42344-de03-11e7-903e-bbf9b8e0ab9b.png)
