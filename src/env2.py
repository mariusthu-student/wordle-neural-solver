from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from random import randint
from features import evalPatterncolor

class WordleEnv(gym.Env):

    def __init__(self, word_list):
        super().__init__()
        self.ListDB = word_list
        self.iteration = -1
        self.answer_word = 'pizza'
        self.answer_int = -1
        self.word_memory = ('pizza',[0,0,0,0,0])

        self.observation_space = spaces.Box(
                    low=0,
                    high=1,
                    shape=(6, 5, 29),
                    dtype=np.int8
                )
        
        self.action_space = gym.spaces.Discrete(len(self.ListDB))

    def _get_obs(self):
        return self.observation

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.iteration = 0
        self.answer_int = self.np_random.integers(0, len(self.ListDB)-1, dtype=int)
        self.answer_word = self.ListDB[self.answer_int]
    
        self.observation = np.zeros(
                    (6, 5, 29),
                    dtype=np.int8
                )
                
        return self._get_obs(), self._get_info()

    def render(self):
        print(self.word_memory)
        
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-6022 words)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        
        
        # Map the discrete action to a word
        mot = self.ListDB[action]
        colors = evalPatterncolor(self.answer_word, mot)

        self.word_memory = (mot,colors)

        #ajoute le mot essayé à l'espace d'observation ainsi que sa correction
        for i in range(5):
            self.observation[self.iteration][i] = [int(l == mot[i]) for l in 'abcdefghijklmnopqrstuvwxyz'] + [int(k == colors[i]) for k in range(3)]

        self.iteration+=1
        
        # Check if agent reached the target
        terminated = (action == self.answer_int or self.iteration >= 6)

        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        reward = 100 if action == self.answer_int else -5

        observation = self._get_obs()
        info = self._get_info()

        
    
        return observation, reward, terminated, truncated, info