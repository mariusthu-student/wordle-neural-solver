import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WordleEnv(gym.Env):
    """
    Simulateur de jeu Wordle respectant les règles officielles et conforme à l'API Gymnasium.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    GREY = 0
    YELLOW = 1
    GREEN = 2

    def __init__(
        self,
        word_list: List[str],
        max_attempts: int = 6,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Normalisation de la liste en minuscules
        self.word_list = [w.lower().strip() for w in word_list]
        self.word_set = set(self.word_list)
        self.max_attempts = max_attempts
        self.render_mode = render_mode

        # Espace d'action discret : chaque indice correspond à un mot de self.word_list
        self.action_space = spaces.Discrete(len(self.word_list))

        # Espace d'observation : matrice (max_attempts, 5 positions, 29 canaux)
        # 26 canaux pour les lettres (a-z) + 3 canaux pour la couleur (Gris, Jaune, Vert)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_attempts, 5, 29),
            dtype=np.float32,
        )

        self.target_word: str = ""
        self.target_int: int = -1
        self.attempts: int = 0
        self.history: List[Tuple[str, List[int]]] = []
        self.observation = np.zeros((self.max_attempts, 5, 29), dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """
        Retourne les informations annexes sur l'état du jeu.
        """
        return {
            "attempts": self.attempts,
            "history": list(self.history),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
        target_word: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        super().reset(seed=seed)

        # Mot secret forcé (via options ou paramètre direct) ou tiré aléatoirement
        chosen_target = target_word or (options.get("target_word") if options else None)
        if chosen_target:
            chosen_target = chosen_target.lower().strip()
            if chosen_target not in self.word_set:
                raise ValueError(f"Le mot cible '{chosen_target}' n'est pas dans la liste de mots.")
            self.target_word = chosen_target
            self.target_int = self.word_list.index(chosen_target)
        else:
            # Borne haute len(self.word_list) non incluse -> tire entre 0 et len-1 inclus
            self.target_int = int(self.np_random.integers(0, len(self.word_list)))
            self.target_word = self.word_list[self.target_int]

        self.attempts = 0
        self.history = []
        self.observation = np.zeros((self.max_attempts, 5, 29), dtype=np.float32)

        # On retourne une copie pour éviter le partage de référence mémoire
        return self.observation.copy(), self._get_info()

    def evaluate_guess(self, guess: str) -> List[int]:
        """
        Calcule le feedback (Vert, Jaune, Gris) pour un mot proposé.
        """
        guess = guess.lower()
        feedback = [self.GREY] * 5
        target_chars = list(self.target_word)

        # 1. Lettres vertes (bonne lettre à la bonne place)
        for i in range(5):
            if guess[i] == self.target_word[i]:
                feedback[i] = self.GREEN
                target_chars[i] = None

        # 2. Lettres jaunes (bonne lettre mais mauvaise place)
        for i in range(5):
            if feedback[i] != self.GREEN:
                char = guess[i]
                if char in target_chars:
                    feedback[i] = self.YELLOW
                    target_chars[target_chars.index(char)] = None

        return feedback

    def _encode_guess(self, guess: str, feedback: List[int]) -> np.ndarray:
        """
        Encode un mot et son feedback en un tenseur (5, 29) de float32.
        """
        row = np.zeros((5, 29), dtype=np.float32)
        for pos in range(5):
            letter_idx = ord(guess[pos]) - ord("a")
            if 0 <= letter_idx < 26:
                row[pos, letter_idx] = 1.0
            color = feedback[pos]
            if 0 <= color < 3:
                row[pos, 26 + color] = 1.0
        return row

    def step(
        self, action: Union[int, np.integer, str]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute un tour de jeu (un coup).
        Accepte soit l'index de l'action discrète, soit directement un mot en str.
        """
        if self.attempts >= self.max_attempts:
            raise RuntimeError("La partie est terminée. Appelez reset() avant de rejouer.")

        # Prise en charge des actions entières (RL), tableaux NumPy scalaires, ou chaîne directe (tests)
        if isinstance(action, np.ndarray) and action.size == 1:
            action = int(action.item())

        if isinstance(action, (int, np.integer)):
            action = int(action)
            if not (0 <= action < len(self.word_list)):
                raise ValueError(
                    f"Action invalide : {action}. Doit être compris entre 0 et {len(self.word_list) - 1}."
                )
            guess = self.word_list[action]
        elif isinstance(action, str):
            guess = action.lower().strip()
            if guess not in self.word_set:
                raise ValueError(f"Mot invalide : '{guess}' n'est pas dans le dictionnaire.")
        else:
            raise TypeError(f"Type d'action non supporté : {type(action)}")

        feedback = self.evaluate_guess(guess)
        self.observation[self.attempts] = self._encode_guess(guess, feedback)
        self.attempts += 1
        self.history.append((guess, feedback))

        won = (guess == self.target_word)
        terminated = won or (self.attempts >= self.max_attempts)
        truncated = False

        # Calcul de la récompense
        if won:
            reward = 10.0 + (self.max_attempts - self.attempts) * 2.0
        elif terminated:
            reward = -10.0
        else:
            reward = -1.0

        info = self._get_info()
        info["won"] = won
        info["target"] = self.target_word if terminated else None
        info["guess"] = guess
        info["feedback"] = feedback

        if self.render_mode == "human":
            self.render()

        # On retourne une copie de l'observation
        return self.observation.copy(), float(reward), terminated, truncated, info

    def render(self):
        """
        Affiche le dernier coup sous forme d'émojis Wordle.
        """
        if self.history:
            last_guess, last_feedback = self.history[-1]
            colors_map = {self.GREY: "⬜", self.YELLOW: "🟨", self.GREEN: "🟩"}
            colors_str = "".join([colors_map.get(c, "⬜") for c in last_feedback])
            output = f"Essai {self.attempts}/{self.max_attempts} : {last_guess.upper()} -> {colors_str}"
            if self.render_mode == "ansi":
                return output
            print(output)
