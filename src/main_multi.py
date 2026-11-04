import os
import shutil
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env4 import WordleEnv

#Pour avoir le visuel : "tensorboard --logdir current_models/tensorboard_logs" dans un 2nd terminal
#

# Configuration
RESET_PROGRESS = False        # Mettre à True pour réinitialiser complètement l'entraînement et repartir de zéro
TOTAL_TIMESTEPS = 500_000     # Nombre de pas à ajouter à cet entraînement (modifiable)
EVAL_FREQ_STEPS = 1_000      # Fréquence d'évaluation (en nombre total de pas)
N_EVAL_EPISODES = 10          # Nombre de parties pour tester le niveau de l'agent
ALLOWED_ATTEMPTS = 30         # Nombre d'essais autorisés par partie (identique à main)
LEARNING_RATE = 0.0003
GAMMA = 0.99
MODELS_DIR = "current_models"
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
LATEST_MODEL_PATH = os.path.join(MODELS_DIR, "ppo_mask_wordle.zip")
LOGS_DIR = os.path.join(MODELS_DIR, "eval_logs")
TENSORBOARD_LOGS_DIR = os.path.join(MODELS_DIR, "tensorboard_logs")


def main():
    # 1. Chargement du dictionnaire de mots
    with open('data/raw/mots.txt', 'r', encoding="utf8") as DBtxt:
        ListDB = [e.strip("'") for e in DBtxt.readline().split(sep=', ')]

    # 2. Préparation des dossiers et détection des cœurs CPU
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOGS_DIR, exist_ok=True)
    num_cpu = min(8, os.cpu_count() or 4)
    print(f"🚀 Lancement de l'entraînement parallélisé sur {num_cpu} cœurs CPU (avec masquage des mots)...")

    # Vérification de la disponibilité de TensorBoard
    try:
        import tensorboard
        tb_log = TENSORBOARD_LOGS_DIR
        print(f"📈 Suivi en temps réel : tensorboard --logdir {TENSORBOARD_LOGS_DIR}")
    except ImportError:
        tb_log = None
        print("⚠️ TensorBoard n'est pas détecté. Entraînement sans logs TensorBoard.")

    # 3. Création des environnements parallélisés avec masquage d'actions
    vec_env = make_vec_env(
        WordleEnv,
        n_envs=num_cpu,
        env_kwargs={"word_list": ListDB, "allowed_attempt": ALLOWED_ATTEMPTS},
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method='spawn')  # Indispensable pour la stabilité sous macOS
    )

    # 4. Environnement d'évaluation dédié avec masquage
    eval_env = Monitor(WordleEnv(word_list=ListDB, allowed_attempt=ALLOWED_ATTEMPTS))

    # Callback qui évalue régulièrement l'agent (en respectant les masques) et sauvegarde le meilleur modèle
    eval_callback = MaskableEvalCallback(
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
        print("🧹 Réinitialisation demandée : suppression des anciens modèles et logs...")
        if os.path.exists(BEST_MODEL_PATH):
            os.remove(BEST_MODEL_PATH)
        if os.path.exists(LATEST_MODEL_PATH):
            os.remove(LATEST_MODEL_PATH)
        shutil.rmtree(TENSORBOARD_LOGS_DIR, ignore_errors=True)
        os.makedirs(TENSORBOARD_LOGS_DIR, exist_ok=True)
        print("🌱 Initialisation d'un nouveau modèle MaskablePPO vierge...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tensorboard_log=tb_log
        )
        reset_timesteps = True
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"📦 Chargement du meilleur modèle historique ({BEST_MODEL_PATH})...")
        model = MaskablePPO.load(BEST_MODEL_PATH, env=vec_env, tensorboard_log=tb_log)
        reset_timesteps = False
    elif os.path.exists(LATEST_MODEL_PATH):
        print(f"📦 Chargement du dernier modèle sauvegardé ({LATEST_MODEL_PATH})...")
        model = MaskablePPO.load(LATEST_MODEL_PATH, env=vec_env, tensorboard_log=tb_log)
        reset_timesteps = False
    else:
        print("🌱 Aucun modèle existant. Initialisation d'un nouveau modèle MaskablePPO...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tensorboard_log=tb_log
        )
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
        model = MaskablePPO.load(BEST_MODEL_PATH)
    else:
        print("\n⚠️ Pas de meilleur modèle distinct trouvé, utilisation du modèle final.")

    # 9. Démonstration des parties jouées par l'agent avec blocage des mots
    print("\n--- Démonstration des parties jouées par l'agent ---")
    demo_env = WordleEnv(word_list=ListDB, allowed_attempt=ALLOWED_ATTEMPTS)
    episodes = 20

    for ep in range(episodes):
        print(f"\nÉpisode {ep + 1} -----------")
        obs, info = demo_env.reset()
        done = False
        while not done:
            action_masks = demo_env.action_masks()
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            demo_env.render()
            done = terminated or truncated


if __name__ == '__main__':
    main()
