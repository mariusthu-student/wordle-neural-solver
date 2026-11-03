import numpy.random as rd

class WordleEnv:
    """
    Simulateur de jeu Wordle respectant les règles officielles.
    """

    GREY = 0
    YELLOW = 1
    GREEN = 2

    def __init__(self, word_list: List[str], max_attempts: int = 6):
        """
        Initialise l'environnement
        """

        self.word_list = word_list
        self.max_attempts = max_attempts
        self.word_set = set(self.word_list) #permet askip de tester la présence d'un mot en O(1)

        self.target_word: str = "" #mot qu'on vise
        self.attempts: int = 0 #nombre d'essais qui ont été faits
        self.history: List[Tuple[str, List[int]]] = []  #exemple d'un élément ("POMME", [0, 0, 0, 0, 1])
        self.is_over: bool = False

        self.reset() #lance la fonction reset

    def reset(self, target_word: Optional[str] = None) -> Dict[str, Any]:

        if target_word:
            target_word = target_word.upper()
            if target_word not in self.word_set:
                raise ValueError(f"Le mot cible '{target_word}' n'est pas dans la liste de mots.")
            self.target_word = target_word
        else:
            self.target_word=str(rd.choice(self.word_list))

        return self._get_state()

    def evaluate_guess(self, guess: str) -> List[int]:
        """
        Calcule le feedback (Vert, Jaune, Gris) pour un mot proposé.
        """
        guess = guess.upper()
        feedback = [self.GREY] * 5 #initialise à 0 0 0 0 0
        target_chars = list(self.target_word)

        #PARTIE 1 : Lettres vertes
        for i in range(5):
            if guess[i] == self.target_word[i]:
                feedback[i] = self.GREEN
                target_chars[i] = None #Lettre retirée de la liste

        for i in range(5):
            if feedback[i] != self.GREEN: #Si on l'a pas déjà comptée
                char = guess[i]
                if char in target_chars: #Si la lettre est encore disponible dans les lettres cibles non consommées
                    feedback[i] = self.YELLOW
                    target_chars[target_chars.index(char)] = None  #On consomme la première occurrence trouvée

        return feedback

    def step(self, guess: str) -> Tuple[List[int], float, bool, Dict[str, Any]]:
        """
        Exécute un tour de jeu (un coup).
        """
        guess = guess.upper()

        if self.is_over:
            raise RuntimeError("La partie est terminée. Appelez reset() avant de rejouer.")

        if guess not in self.word_set:
            raise ValueError(f"Mot invalide : '{guess}' n'est pas dans le dictionnaire.")

        # Calcul des couleurs
        feedback = self.evaluate_guess(guess)
        self.attempts += 1

        won = (guess == self.target_word)
        self.is_over = won or (self.attempts >= self.max_attempts)

        # Sauvegarde dans l'historique
        self.history.append((guess, feedback))

        # CALCUL DE LA RÉCOMPENSE (Pour l'apprentissage par renforcement)
        if won:
            # Récompense maximale si victoire, avec bonus si réussi en peu de coups
            reward = 10.0 + (self.max_attempts - self.attempts) * 2.0
        elif self.is_over:
            # Pénalité en cas d'échec (6 coups épuisés sans trouver)
            reward = -10.0
        else:
            # Petite pénalité à chaque coup pour inciter à résoudre rapidement
            reward = -1.0

        info = {
            "won": won,
            "target": self.target_word if self.is_over else None,
            "attempts": self.attempts
        }

        return feedback, reward, self.is_over, info

    def _get_state(self):
        """
        Retourne l'état actuel.
        """
        return {
            "attempts": self.attempts,
            "history": self.history,
            "is_over": self.is_over
        }