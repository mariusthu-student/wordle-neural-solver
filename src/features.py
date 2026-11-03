import numpy as np

def char_to_index(char: str) -> int:
    """
    Convertit un caractère (A-Z) en son indice (0-25).
    Exemple: 'A' -> 0, 'Z' -> 25.
    """
    return ord(char.upper()) - ord('A')


def encode_state(history: list, max_attempts: int = 6) -> np.ndarray:
    """
    Convertit l'historique d'une partie en une matrice numérique.
    """
    # Matrice remplie de 0 : 6 essais, 5 positions, 29 canaux (26 pour les lettres + 3 pour les couleurs)
    state = np.zeros((max_attempts, 5, 26 + 3), dtype=np.float32)

    for attempt_idx, (guess, feedback) in enumerate(history):
        if attempt_idx >= max_attempts:
            break
            
        for pos_idx in range(5):
            char = guess[pos_idx]
            color_code = feedback[pos_idx]
            
            # 1. Encodage One-Hot de la lettre (canaux 0 à 25)
            letter_idx = char_to_index(char)
            state[attempt_idx, pos_idx, letter_idx] = 1.0
            
            # 2. Encodage One-Hot de la couleur (canaux 26 à 28)
            # color_code: 0 (Gris) -> canal 26, 1 (Jaune) -> canal 27, 2 (Vert) -> canal 28
            state[attempt_idx, pos_idx, 26 + color_code] = 1.0

    return state