# Preliminary Baba Is You Reinforcement Learning

## Introduction

[Baba Is You](https://store.steampowered.com/app/736260/Baba_Is_You/) is a game where the rules constantly change. It's an interesting challenge for reinforcement learning because the agent must reason about the meaning of the words and not simply memorize input patterns. For example, if the rules are "Baba is you. Flag is win. Skull is lose.", then the agent should learn to move the Baba object to the Flag object and avoid the Skull object. Now, if the rules change to "Baba is you. Flag is lose. Skull is win.", the previously learned policy would result in defeat, even though the game states look almost identical.

This project focuses on such an introductory scenario in the world of Baba Is You. Each level is a small grid containing 3 rules (for example, "a is you", "b is win", "c is lose"), and 3 objects (A, B, and C). The goal is to move the "you" object to the "win" object and avoid the "lose" object. Even this simple task is difficult for current deep reinforcement learning methods; more innovation will be needed to play more advanced levels, where, for example, the player must change the rules.

