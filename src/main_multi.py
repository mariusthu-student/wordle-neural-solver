import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import WordleEnv

def main():
    with open('data/raw/mots.txt', 'r', encoding="utf8") as DBtxt:
        ListDB = [e.strip("'") for e in DBtxt.readline().split(sep=', ')]

    num_cpu = min(8, os.cpu_count())
    print(f"Lancement de l'entraînement parallélisé sur {num_cpu} cœurs CPU...")

    vec_env = make_vec_env(
        WordleEnv,
        n_envs=num_cpu,
        env_kwargs={"word_list": ListDB},
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')  # Indispensable pour la stabilité sur macOS
    )

    model = A2C('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=20_000_000)

    vec_env.close()

    #Démonstration / Évaluation sur un environnement individuel
    print("\n--- Démonstration des parties jouées par l'agent ---")
    eval_env = WordleEnv(word_list=ListDB)
    episodes = 5

    for ep in range(episodes):
        print(f"\nÉpisode {ep + 1} -----------")
        obs, info = eval_env.reset()
        done = False
        while not done:
            # deterministic=True permet à l'agent d'exploiter ses meilleures prédictions
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            eval_env.render()
            done = terminated or truncated

        if info["won"]:
            print(f"🎉 Trouvé en {info['attempts']} essais !")
        else:
            print(f"❌ Échec ! Le mot était : {info['target'].upper()}")


if __name__ == '__main__':
    main()