# Preliminary Baba Is You Reinforcement Learning

## Introduction

[Baba Is You](https://store.steampowered.com/app/736260/Baba_Is_You/) is a game where the rules constantly change. It's an interesting challenge for reinforcement learning because the agent must reason about the meaning of the words and not simply memorize input patterns. For example, if the rules are "Baba is you. Flag is win. Skull is lose.", then the agent should learn to move the Baba object to the Flag object and avoid the Skull object. Now, if the rules change to "Baba is you. Flag is lose. Skull is win.", the previously learned policy would result in defeat, even though the game states look almost identical. The policy must be based on the *roles* of the objects rather than the objects themselves.

This project focuses on such an introductory scenario in the world of Baba Is You. Each level is a small grid containing 3 rules (for example, "a is you", "b is win", "c is lose"), and 3 objects (A, B, and C). The goal is to move the "you" object to the "win" object and avoid the "lose" object. Even this simple task is difficult for current deep reinforcement learning methods. More innovation will be needed to play more advanced levels, where, for example, the player must change the rules.

## External Rule Task

This first attempt is based on [Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning](https://proceedings.mlr.press/v139/hanjie21a.html), in which a rulebook describes the role of each object in a grid. The goal is to convert the object representation into a role representation, so that reinforcement learning can be based on the roles of the objects rather than the objects themselves. Applying these methods to Baba Is You, the rules become key:value pairs of the form "a:you, b:win, c:lose". For each object A, B, C, etc., a *query* must be learned to match the object to the associated key, for example object "A" to symbol "a". Then, an [Attention](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) layer returns the value/role for each query, so "A" becomes "you". Once the objects are replaced by their roles, regular reinforcement learning may proceed with the goal of always moving "you" to "win". Figure 1 illustrates the process.

![External Rule Model](https://github.com/uvulab/baba_rl_intro/blob/main/external.png)
**Figure 1: External Rule Model. Inputs are on the top row: grid/objects, keys/symbols, and values/roles. An attention layer converts each object to its role.**

## Internal Rule Task

The external rule task is easier to train and useful for choosing the right reinforcement learning parameters; but for actually playing Baba Is You, a different approach is needed, because the rules exist inside the grid and can be changed by the player. We now attempt to put the rules inside the grid and extract the same key:value pairs that we started with before.
