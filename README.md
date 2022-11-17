# AI_project_splendor_game
## Project overview
We have produced two different AI agents based on the **A star and alpha-beta pruning algorithms** to play this board game and we will introduce and analyze their performance here. The implementation of these two algorithms is different, but they both get good results.
# Design choice
In this contest, we chose two different methods: A star and alpha-beta pruning algorithms.
## AI Method 1 - A star
The A-Star algorithm consists of a priority queue to choose the highest priority action, an evaluation function of states to analyze the board condition, an evaluation function of each action, and a heuristic function to assess the action.

P.s. Due to many limitations, although we hope to implement the A star algorithm, the final implementation is not complete. Thus this algorithm is actually more like GBFS.
## AI Method 2 - Alpha-beta Pruning
This algorithm adopts roughly the same evaluation method as A Star, but is quite different in algorithm structure. Its core methods are a maximum pruning function and a minimum pruning function, which is corresponding to choose the most optimum action to our AI agent self and most optimum action to its opponent (the worst action to our agent) respectively.
## Rank
7/87
