# SigmaGeek Stone Automata Maze Challenge

## Intro

Repository with my submission to the Stone Automata Maze Challenge, hosted by SigmaGeek, with which I obtained the 4th place ([results](https://sigmageek.com/stone_results/stone-automata-maze-challenge) and [complete call](https://sigmageek.com/challenge/stone-automata-maze-challenge)).

## Folders

* challenges: files with the statements of the challenges
* submissions: submitted files
* src: latest files, with improvements done after submission
* images: animated gifs of some solutions

## Submission

My submitted results:

* challenge 1: 6176 steps (optimal)
* challenge 2: 6016 steps (optimal)
* challenge 3: 6200 steps (trivial, without using the given ability)
* challenge 4: closest distance 2699 on epoch 2299 (optimal)
* challenge 5: 3095 particles (execution interrupted due to the deadline)

## Solver

The most recent solver for challenges 1 to 4 is found [here](src/solver.py). It finds the optimal solutions for challenges 1, 2 and 4. Also, it finds the trivial solution for challenge 3 (shortest path without using the ability to change the state of the cells).

The most recent solver for challenge 5 is found [here](src/solver_challenge_5.py).

#### Running the solver

To run the solver for challenges 1 to 4, use the command

```python3 solver.py N```

where N is the number of the desired challenge (1-4).

To run the solver for challenge 5, just execute the script without parameters

```python3 solver_challenge_5.py```

#### Actual results

* challenges 1 to 4: same as the submitted ones
* challenge 5: 10165 particles (20 more than the first place solution! :star2:)

#### Elapsed times :hourglass_flowing_sand: (load maze states + find solution)

* challenge 1: 137 s + 36 s
* challenge 2: 125 s + 222 s
* challenge 3: 126 s + 33 s 
* challenge 4: 130 s + 172 s
* challenge 5: 109959 s (1 day, 6h, 32min, 39s :zzz:)
