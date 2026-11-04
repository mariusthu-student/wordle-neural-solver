import os
from sb3_contrib import MaskablePPO
import numpy as np
from sys import stdout

from env4 import WordleEnv

MODELS_DIR = "current_models"
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
ALLOWED_ATTEMPTS = 30
NB_ITERATION = 1000

def main():
    # 1. Chargement du dictionnaire de mots
    with open('data/raw/mots.txt', 'r', encoding="utf8") as DBtxt:
        ListDB = [e.strip("'") for e in DBtxt.readline().split(sep=', ')]

    if os.path.exists(BEST_MODEL_PATH):
        print(f"\n🏆 Chargement du meilleur modèle sauvegardé ({BEST_MODEL_PATH}) pour la démonstration...")
        model = MaskablePPO.load(BEST_MODEL_PATH)
    else:
        print("\n⚠️ Pas de meilleur modèle distinct trouvé, vérifier le chemin d'accès")
        return
        
    demo_env = WordleEnv(word_list=ListDB, allowed_attempt=ALLOWED_ATTEMPTS)

    perfs = []
    
    for ep in range(NB_ITERATION):
        obs, _ = demo_env.reset()
        done = False
        while not done:
            action_masks = demo_env.action_masks()
            action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, _, terminated, truncated, _ = demo_env.step(action)
            done = terminated or truncated
        perfs.append(demo_env.iteration)
        pourc = int((ep+1) / NB_ITERATION * 100)
        stdout.write('\r[' + '█' * int(pourc/2) + '░' * int((100 - pourc)/2) + f'] tâche {pourc}% complète :')
    print(f'\n{NB_ITERATION} essais réalisés')

    perfs = np.array(perfs)
    win = np.sum([e<=6 for e in perfs])
    print(f"| coups moyen :{np.mean(perfs)} | winrate : {win /NB_ITERATION} | variance :{np.var(perfs)} | écart-type :{np.std(perfs)}")


if __name__ == '__main__':
    main()
