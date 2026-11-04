import numpy as np

with open('data/raw/mots.txt','r', encoding="utf8") as DBtxt:
    ListDB = [e.strip("'") for e in DBtxt.readline().split(sep=', ')]


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

def evalPatterncolor(toGuess, mot):
    taillemot = len(mot)
    pattern = []
    vertes, oranges = '', []
    for i in range(taillemot):
        l = mot[i]
        if l == toGuess[i]:
            pattern.append(2)
            vertes+=l
        elif l not in toGuess:
            pattern.append(0)
        else:
            pattern.append(1)
            oranges.append((l,i))

    letavailable = {}
    for e in toGuess:
        letavailable[e] = letavailable[e] + 1 if e in letavailable else 1

    for l in vertes:
        letavailable[l] -= 1

    for (l, i) in oranges:  # on souhaite éliminer les faux oranges
        if letavailable[l] <= 0:
            pattern[i] = 0
        letavailable[l] -= 1
    return pattern

def matchInList(patmot, pattern):
    rouge,vert,orange = [],[],[]
    for i in range(len(pattern)):
        sym = pattern[i]
        if sym == 2:
            vert.append((patmot[i], i))
        if sym == 0:
            rouge.append((patmot[i], i))
        if sym == 1:
            orange.append((patmot[i], i))

    let_connu = {}
    for (l,_) in orange + vert:
        let_connu[l] = 1 if l not in let_connu else let_connu[l] + 1

    DBclean = []
    for mot in ListDB:
        let_mot = {}
        for l in mot:
            let_mot[l] = let_mot[l]+1 if l in let_mot else 1
        toKeep = True
        for (l,_) in rouge:
            if l not in [e[0] for e in orange] and l not in [e[0] for e in vert]:
                if l in mot:
                    toKeep = False                        #print("il y a une rouge qui n'est pas pas verte ni orange, présente ds le mot")
                    break
            elif l in let_mot and let_mot[l] != let_connu[l]:
                toKeep = False                    #print("magie de Jean")
                let_mot[l]-=1
                break
            elif l not in let_mot:
                toKeep = False
                break
        for (l,i) in vert:
            if mot[i] != l:
                toKeep = False                    #print("manque une verte à son indice")
                break
        for (l,i) in orange:
            if mot[i] == l:
                toKeep = False                    #print("une orange n'as pas bougé")
                break
            if l not in mot or let_mot[l]<let_connu[l]:
                toKeep = False                    #print("il manque une orange")
                break
        if toKeep:
            DBclean.append(mot)
    return DBclean

