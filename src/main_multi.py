import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from env import WordleEnv

# Configuration
RESET_PROGRESS = False     # Mettre à True pour réinitialiser complètement l'entraînement et repartir de zéro
TOTAL_TIMESTEPS = 1_000_000  # Nombre de pas à ajouter à cet entraînement (modifiable)
EVAL_FREQ_STEPS = 10_000   # Fréquence d'évaluation (en nombre total de pas)
N_EVAL_EPISODES = 30       # Nombre de parties pour tester le niveau de l'agent
MODELS_DIR = "models"
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "a2c_wordle.zip")
LOGS_DIR = os.path.join(MODELS_DIR, "eval_logs")


def main():
    # 1. Chargement du dictionnaire de mots
    with open('data/raw/mots.txt', 'r', encoding="utf8") as DBtxt:
        ListDB = [e.strip("'") for e in DBtxt.readline().split(sep=', ')]

    # 2. Préparation des dossiers et détection des cœurs CPU
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    num_cpu = min(8, os.cpu_count() or 4)
    print(f"🚀 Lancement de l'entraînement parallélisé sur {num_cpu} cœurs CPU...")

    # 3. Création des environnements parallélisés
    vec_env = make_vec_env(
        WordleEnv,
        n_envs=num_cpu,
        env_kwargs={"word_list": ListDB},
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')  # Indispensable pour la stabilité sous macOS
    )

    # 4. Environnement d'évaluation dédié
    eval_env = Monitor(WordleEnv(word_list=ListDB))

    # Callback qui évalue régulièrement l'agent et sauvegarde UNIQUEMENT s'il bat son record
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOGS_DIR,
        eval_freq=max(1, EVAL_FREQ_STEPS // num_cpu),
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1
    )

    # 5. Chargement (priorité au meilleur modèle existant) ou réinitialisation
    if RESET_PROGRESS:
        print("🧹 Réinitialisation demandée : suppression des anciens modèles...")
        if os.path.exists(BEST_MODEL_PATH):
            os.remove(BEST_MODEL_PATH)
        if os.path.exists(LATEST_MODEL_PATH):
            os.remove(LATEST_MODEL_PATH)
        print("🌱 Initialisation d'un nouveau modèle vierge...")
        model = A2C('MlpPolicy', vec_env, verbose=1)
        reset_timesteps = True
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"📦 Chargement du meilleur modèle historique ({BEST_MODEL_PATH})...")
        model = A2C.load(BEST_MODEL_PATH, env=vec_env)
        reset_timesteps = False
    elif os.path.exists(LATEST_MODEL_PATH):
        print(f"📦 Chargement du dernier modèle sauvegardé ({LATEST_MODEL_PATH})...")
        model = A2C.load(LATEST_MODEL_PATH, env=vec_env)
        reset_timesteps = False
    else:
        print("🌱 Aucun modèle existant. Initialisation d'un nouveau modèle...")
        model = A2C('MlpPolicy', vec_env, verbose=1)
        reset_timesteps = False

    # 6. Entraînement (cumulatif si reprise, remis à zéro si réinitialisation)
    print(f"⏳ Début de l'entraînement pour {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        reset_num_timesteps=reset_timesteps
    )

    # 7. Sauvegarde du modèle final actuel
    model.save(LATEST_MODEL_PATH.replace('.zip', ''))
    print(f"💾 Dernier état sauvegardé dans {LATEST_MODEL_PATH}")
    vec_env.close()

    # 8. Rechargement du MEILLEUR modèle pour la démonstration (évite d'utiliser un modèle ayant régressé)
    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n🏆 Chargement du meilleur modèle sauvegardé ({BEST_MODEL_PATH}) pour la démonstration...")
        model = A2C.load(BEST_MODEL_PATH)
    else:
        print("\n⚠️ Pas de meilleur modèle distinct trouvé, utilisation du modèle final.")

    # 9. Démonstration des parties jouées par l'agent
    print("\n--- Démonstration des parties jouées par l'agent ---")
    demo_env = WordleEnv(word_list=ListDB)
    episodes = 5

    for ep in range(episodes):
        print(f"\nÉpisode {ep + 1} -----------")
        obs, info = demo_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            demo_env.render()
            done = terminated or truncated

        if info["won"]:
            print(f"🎉 Trouvé en {info['attempts']} essais !")
        else:
            print(f"❌ Échec ! Le mot était : {info['target'].upper()}")


if __name__ == '__main__':
    main()
