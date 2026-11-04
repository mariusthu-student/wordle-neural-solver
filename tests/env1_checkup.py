from src.env import WordleEnv

# Simulation rapide avec une petite liste de test
word_list = ["POMME", "ROUGE", "TARTE", "LIVRE", "ARBRE"]
env = WordleEnv(word_list)

# Forcer un mot secret pour tester
env.reset(target_word="ROUGE")

# Jouer un premier coup
obs, reward, terminated, truncated, info = env.step("POMME")
print("Coup 1 'POMME' -> Feedback:", info["feedback"])  # Attendu: [0, 2, 0, 0, 2]

# Jouer le mot gagnant
obs, reward, terminated, truncated, info = env.step("ROUGE")
print("Coup 2 'ROUGE' -> Feedback:", info["feedback"])  # Attendu: [2, 2, 2, 2, 2]
print("Victoire ?", info["won"], "| Récompense :", reward)