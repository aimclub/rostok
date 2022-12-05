from gymnasium.envs.registration import register

register(
     id="gym_rostok/GraphGrammarEnv-v0",
     entry_point="rostok.gym_rostok.envs:GraphGrammarEnv",
     max_episode_steps=300,
)

register(
     id="gym_rostok/GGrammarControlOpimizingEnv-v0",
     entry_point="rostok.gym_rostok.envs:GGrammarControlOpimizingEnv",
     max_episode_steps=300,
)
