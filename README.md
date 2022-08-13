# Preliminary Baba Is You Reinforcement Learning

## Introduction

[Baba Is You](https://store.steampowered.com/app/736260/Baba_Is_You/) is a game where the rules constantly change. It's an interesting challenge for reinforcement learning because the agent must reason about the meaning of the words and not simply memorize input patterns. For example, if the rules are "Baba is you. Flag is win. Skull is lose.", then the agent should learn to move the Baba object to the Flag object and avoid the Skull object. Now, if the rules change to "Baba is you. Flag is lose. Skull is win.", the previously learned policy would result in defeat, even though the game states look almost identical. The policy must be based on the *roles* of the objects rather than the objects themselves.

This project focuses on such an introductory scenario in the world of Baba Is You. Each level is a small grid containing 3 rules (for example, "a is you", "b is win", "c is lose"), and 3 objects (A, B, and C). The goal is to move the "you" object to the "win" object and avoid the "lose" object. Even this simple task is difficult for current deep reinforcement learning methods. More innovation will be needed to play more advanced levels, where, for example, the player must change the rules.

## External Rule Task

This first attempt is based on [Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning](https://proceedings.mlr.press/v139/hanjie21a.html), in which a rulebook describes the role of each object in a grid. The goal is to convert the object representation into a role representation, so that reinforcement learning can be based on the roles of the objects rather than the objects themselves. Applying these methods to Baba Is You, the rules become key:value pairs of the form "a:you, b:win, c:lose". For each object A, B, C, etc., a *query* must be learned to match the object to the associated key, for example object "A" to name "a". Then, an [Attention](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) layer returns the value/role for each query, so "A" becomes "you". Once the objects are replaced by their roles, regular reinforcement learning may proceed with the goal of always moving "you" to "win". Figure 1 illustrates the process.

![External Rule Model](https://github.com/uvulab/baba_rl_intro/blob/main/external.png)
**Figure 1: External Rule Model. Inputs are on the top row: grid/objects, keys/names, and values/roles. An attention layer converts each object to its role.**

## Internal Rule Task

The external rule task is easier to train and useful for choosing the right reinforcement learning parameters; but for actually playing Baba Is You, a different approach is needed, because the rules exist inside the grid and can be changed by the player. (In this task, the rules still cannot change because they're on the edge of the grid, but changing rules will be necessary in the future.) We now attempt to put the rules inside the grid and extract the same key:value pairs that we started with before. Figure 2 illustrates.

![Internal Rule Model](https://github.com/uvulab/baba_rl_intro/blob/main/internal.png)
**Figure 2: Internal Rule Model. Input grid channels are separated for each category: objects, names, and roles. Queries and values must be learned to perform the same attention step as in the previous task.**

As before, each object must be converted into a query, which matches a provided name/key. The main difference is that a convolutional layer must now be trained to detect roles such as "is you" and place a value representation "you" in the same position as the corresponding key. Ideally, the position of the "a" will then contain a key:value pair ("a:you") which can be queried by the object "A". Note: empty spaces in the grid will contain "nothing:nothing" rules.

Currently, we provide separate input grids for objects, names, and roles. A more challenging version of this problem would have all inputs together, so that the agent must learn the difference between the three categories.

## Training

We use [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) as our reinforcement learning algorithm, based on [this implementation](https://github.com/nikhilbarhate99/PPO-PyTorch). You are welcome to experiment with the parameters to try for better and faster results.

To make sure the model is actually learning to reason about the meaning of "you", "win", and "lose", and not simply memorizing all possible rule combinations, we leave one object-role combination out of training and use it for testing. The agent should be able to generalize to object-role combinations not seen in training.

## Usage

Install tensorflow and numpy.

Create a directory called "models" in the working directory.

Edit the parameters in `main.py` to choose the task (internal or external), grid size, model name, etc.

Run `python main.py`.

You should see the success rate start to improve after several hours. Full training, especially for the internal rule task, can take 24 hours or more, depending on how fast your system is.

You may also run `python manual_test.py` to try the Baba Is You environment.

## Discussion

The external rule task works reliably. The internal rule task works with seed 1, but not seeds 2, 3, and 4; and has worse results when testing on an unseen combination than when training. Clearly, more work is needed to play more advanced Baba Is You levels with AI. At the very least, hierarchical reinforcement learning will be needed if a level requires multiple steps like changing a rule, opening a door, and only then moving to the goal. Reinforcement learning still has so many limitations that an entirely different approach may be needed:

-The agent must be trained on millions of example games, even to learn the meaning of simple concepts like "you", "win", and "lose".

-An agent trained on simple levels cannot automatically generalize to more complicated levels.

-Planning many moves into the future is not possible.

-Training does not scale with grid size, because even with convolution and pooling, larger grids have more state/action/reward combinations to learn.

-Training can fail with no explanation depending on the random seed or hyperparameter choices.

How can we build a more human-like agent that explores the world, builds a conceptual model, continually learns new concepts, and communicates in a way humans can understand?
