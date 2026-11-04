from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from features import evalPatterncolor,matchInList



class WordleEnv(gym.Env):
    """
        Envir n°3 :
            observation space is a 6*5*29 tensor
            reward now take into consideration DB reduction
            reward now blames words reuse
    """
    def __init__(self, word_list, allowed_attempt=6):
        super().__init__()
        self.ListDB = word_list
        self.allowed_attempt = allowed_attempt
        self.iteration = -1
        self.answer_word = 'pizza'
        self.answer_int = -1
        self.word_memory = ('pizza',[0,0,0,0,0])
        self.potential_answer = set(self.ListDB)
        self.reduction_reward_rate = 4

        self.observation_space = spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.allowed_attempt, 5, 29),
                    dtype=np.int8
                )
        
        self.action_space = gym.spaces.Discrete(len(self.ListDB))

    def _get_obs(self):
        return self.observation

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options = None,):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.iteration = 0
        self.potential_answer = set(self.ListDB)
        self.answer_int = self.np_random.integers(0, len(self.ListDB)-1, dtype=int)
        self.answer_word = self.ListDB[self.answer_int]

        self.observation = np.zeros(
                    (self.allowed_attempt, 5, 29),
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
        prev_word,_ = self.word_memory
        self.word_memory = (mot,colors)
        
        
        #ajoute le mot essayé à l'espace d'observation ainsi que sa correction
        for i in range(5):
            self.observation[self.iteration][i] = [int(l == mot[i]) for l in 'abcdefghijklmnopqrstuvwxyz'] + [int(k == colors[i]) for k in range(3)]

        self.iteration+=1
        
        # Check if agent reached the target
        terminated = (action == self.answer_int or self.iteration >= self.allowed_attempt)

        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        reward = 100 if action == self.answer_int else -5

        #reward if word reduces number of answer
        previous_DB_len = len(self.potential_answer)
        self.potential_answer = self.potential_answer & set(matchInList(mot,colors, self.ListDB))
        reduction_reward = previous_DB_len/len(self.potential_answer)*self.reduction_reward_rate
        reward+= reduction_reward

        #reward blame if uses the same word twice
        repetition_blame = (mot == prev_word)*(-1000)
        reward+=repetition_blame

        observation = self._get_obs()
        info = self._get_info()        
    
        return observation.copy(), reward, terminated, truncated, info

if __name__ == "__main__":
    print("imported env3")
    