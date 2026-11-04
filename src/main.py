import gymnasium as gym
from stable_baselines3 import A2C

from env2 import WordleEnv

with open('data/raw/mots.txt','r', encoding="utf8") as DBtxt:
    ListDB = [e.strip("'") for e in DBtxt.readline().split(sep=', ')]

env = WordleEnv(word_list=ListDB)
env.reset()

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)


episodes = 5

for ep in range(episodes):
    print(f'episode : {ep}  -----------')
    print()
    obs,info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, trunc, info = env.step(action)
        env.render()
        print(rewards)
