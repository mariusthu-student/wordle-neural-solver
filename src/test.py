from gymnasium.utils.env_checker import check_env
from env2 import WordleEnv

# This will catch many common issues
try:
    check_env(WordleEnv(['petit', 'pomme', 'rouge']))
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
