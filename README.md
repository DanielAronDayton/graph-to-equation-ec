# Programming Assignment 02 (pa02)
Genetic Programming (GP)

## Background reading:
If you want a review on tree traversals, see my data structures page:
* https://www.cnsr.dev/index_files/Classes/DataStructures/Content.html

For GP background and reading, see:
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/06-PopularVariants.html#genetic-programming-1
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/18-GeneticProgramming.html
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/06-PopularVariants/GP_book1.pdf
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/06-PopularVariants/evolvedtowin.pdf

## Task
Evolve numeric programs that take one integer input and produce one float output.
For more detail, see:
* Top of `gp_lisp.py`
* main function in `gp_lisp.py`
* Tests in `randy_tests.py`

## Goal: evolve arbitrary programs!
You can edit everything except for:
* Representation
* Basic print function
* The fitness function and it's supporting data structures
* Main driver

You must write the rest!
You can use the function headers provided,
but you do not have to.
The only constraint is that you use a custom EC,
that you write yourself.

## Grading
WARNING: the grade you see on git-classes may not be your actual grade.
If not enough people do well,
then I may normalize to the best score, 
such that the best student performance get's 100,
and everyone else whose program actually completes get's an interpolated score from there.

## Mutation and recombination
You need to write GP-friendly recombination and mutation operators.
This is a substantial task!
You will likely need quite a few helper functions, which you can add freely.
In addition to the GP reading above, recall:
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/04-RepresentMutateRecombine.html

## Diversity and population management
Your EC must maintain sufficient diversity,
while at the same time selecting the right individuals.
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/05-FitnessSelection.html

## Tuning, Control, and Meta-optimization
It's very unlikely your EC will find the best solution the first time.
You must tune, or even control your EC's parameters to meta-optimize.
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/07-ParameterTuning.html
* https://www.cnsr.dev/index_files/Classes/EvolutionaryComputation/Content/08-ParameterControl.html
