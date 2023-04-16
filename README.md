# SigmaGeek Stone Automata Maze Challenge

## Intro

Repository with my submission to the Stone Automata Maze Challenge, hosted by SigmaGeek, with which I obtained the 4th place ([results](https://sigmageek.com/stone_results/stone-automata-maze-challenge) and [complete call](https://sigmageek.com/challenge/stone-automata-maze-challenge)).

## Folders

* challenges: files with the statements of the challenges
* submissions: submitted files
* src: latest files, with improvements done after submission
* images: animated gifs of some solutions

## Solver

The most recent solver is found [here](src/solver.py). It finds the optimal solutions for challenges 1, 2 and 4. Also, it finds the optimal trivial solution for challenge 3 (without using the ability to change the state of the cells).

#### Running the solver

Run the command 

```python3 solver.py N```

where N is the number of the desired challenge (1-4).

#### Elapsed times:

* challenge 1: 33.61 s
* challenge 2: 206.37 s
* challenge 3: 34.65 s 
* challenge 4: 33.72 s (until epoch 3999)

An improved solver for challenge 5 will be added in the future.
