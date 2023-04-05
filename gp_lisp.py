#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
From data (input-output pairings),
and a set of operators and operands as the only starting point,
write a program that will evolve programmatic solutions,
which take in inputs and generate outputs.

Each program will have 1 numeric input and 1 numeric output.
This is much like regression in our simple case,
though can be generalized much further,
toward arbitrarily large and complex programs.

This assignment is mostly open-ended,
with a couple restrictions:
"""

# %%
import random
from typing import TypedDict
from typing import Optional
import math
import numpy as np

# import json

# import datetime
# import subprocess


# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
OPERATORS = "+-/*^"
VARIABLES = "x"#"xsgtf"

MAX_ITERATIONS = 2000
MAX_TIME_WITHOUT_IMPROVEMENT = 300
DEFAULT_POP_SIZE = 200
DEFAULT_DEPTH = 2
MUTATE_RATE_STEEPNESS = 10.0
MUTATE_DEPTH = 3
SPONTANEOUS_PERCENTAGE = 0.25
SPONTANEOUS_DEPTH = 4
RECOMBINE_RATE_MULTIPLIER = 0.5
ERROR_PRECISION = 0.01
GENOME_LEN_WEIGHT = 0.2


class Node:
    """
    Example prefix formula:
    Y = ( * ( + 20 45 ) ( - 56 X ) )
    This is it's tree:
       *
      /  \
    +     -
    / \   / \
    20 45 56  X

    root = Node(
        data="*",
        left=Node(data="+", left=Node("20"), right=Node("45")),
        right=Node(data="-", left=Node("56"), right=Node("X")),
    )
    """

    def __init__(
        self, data: str, left: Optional["Node"] = None, right: Optional["Node"] = None
    ) -> None:
        self.data = data
        self.left = left
        self.right = right


class Individual(TypedDict):
    """Type of each individual to evolve"""

    genome: Node
    fitness: float


Population = list[Individual]


class IOpair(TypedDict):
    """Data type for training and testing data"""

    input1: list[float]
    output1: float


IOdata = list[IOpair]


def print_tree(root: Node, indent: str = "") -> None:
    """
    Pretty-prints the data structure in actual tree form.
    >>> print_tree(root=root, indent="")
    """
    if root.right is not None and root.left is not None:
        print_tree(root=root.right, indent=indent + "    ")
        print(indent, root.data)
        print_tree(root=root.left, indent=indent + "    ")
    else:
        print(indent + root.data)


def parse_expression(source_code: str) -> Node:
    """
    Turns prefix code into a tree data structure.
    >>> clojure_code = "( * ( + 20 45 ) ( - 56 X ) )"
    >>> root = parse_expression(clojure_code)
    """
    source_code = source_code.replace("(", "")
    source_code = source_code.replace(")", "")
    code_arr = source_code.split()
    return _parse_experession(code_arr)


def _parse_experession(code: list[str]) -> Node:
    """
    The back-end helper of parse_expression.
    Not intended for calling directly.
    Assumes code is prefix notation lisp with space delimeters.
    """
    if code[0] in OPERATORS:
        return Node(
            data=code.pop(0),
            left=_parse_experession(code),
            right=_parse_experession(code),
        )
    else:
        return Node(code.pop(0))


def parse_tree_print(root: Node) -> None:
    """
    Stringifies to std-out (print) the tree data structure.
    >>> parse_tree_print(root)
    """
    if root.right is not None and root.left is not None:
        print(f"( {root.data} ", end="")
        parse_tree_print(root.left)
        parse_tree_print(root.right)
        print(") ", end="")
    else:
        # for the case of literal programs... e.g., `4`
        print(f"{root.data} ", end="")


def parse_tree_return(root: Node) -> str:
    """
    Stringifies to the tree data structure, returns string.
    >>> stringified = parse_tree_return(root)
    """
    if root.right is not None and root.left is not None:
        return f"( {root.data} {parse_tree_return(root.left)} {parse_tree_return(root.right)} )"
    else:
        # for the case of literal programs... e.g., `4`
        return root.data


def initialize_individual(genome: str, fitness: float) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as Node, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[Node, int]
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    >>> ind1 = initialize_individual("( + ( * C ( / 9 5 ) ) 32 )", 0)
    """
    return {"genome": parse_expression(genome), "fitness": fitness}


def initialize_data(input1: list[float], output1: float) -> IOpair:
    """
    For mypy...
    """
    return {"input1": input1, "output1": output1}


def prefix_to_infix(prefix: str) -> str:
    """
    My minimal lisp on python interpreter, lol...
    >>> C = 0
    >>> print(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )"))
    >>> print(eval(prefix_to_infix("( + ( * C ( / 9 5 ) ) 32 )")))
    """
    prefix = prefix.replace("(", "")
    prefix = prefix.replace(")", "")
    prefix_arr = prefix.split()
    stack = []
    i = len(prefix_arr) - 1
    while i >= 0:
        if prefix_arr[i] not in OPERATORS:
            stack.append(prefix_arr[i])
            i -= 1
        else:
            str = "(" + stack.pop() + prefix_arr[i] + stack.pop() + ")"
            stack.append(str)
            i -= 1
    return stack.pop()


def populate_with_variables(formula: str) -> str:
    formula_arr = formula.split()
    for var in VARIABLES:
        count = 0
        while var == "x" or count < 50:
            i = random.randint(0, len(formula_arr) - 1)
            if formula_arr[i] not in OPERATORS and formula_arr[i] not in VARIABLES:
                formula_arr[i] = var
                break
            count += 1
    return " ".join(formula_arr)


def gen_rand_prefix_code(depth_limit: int, rec_depth: int = 0, var_chance: float = 0) -> str:
    """
    Generates one small formula,
    from OPERATORS and ints from -100 to 200
    """
    rec_depth += 1
    if rec_depth < depth_limit:
        if random.random() < 0.8:
            return (
                random.choice(OPERATORS)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth, var_chance)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth, var_chance)
            )
        elif random.random() < var_chance:
            return "x"
        else:
            return str(random.uniform(-20.0, 20.0))
    elif random.random() < var_chance:
        return "x"
    else:
        return str(random.uniform(-20.0, 20.0))





def initialize_pop(pop_size: int, depth: int = 2) -> Population:
    """
    Purpose:        Create population to evolve
    Parameters:     Goal string, population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          random.choice-1, string.ascii_letters-1, initialize_individual-n
    Example doctest:
    """
    pop = []
    for i in range(pop_size):
        genome = gen_rand_prefix_code(depth_limit=DEFAULT_DEPTH)
        genome = populate_with_variables(genome)
        # for j in range(len(example_genome)):
        #    genome += random.choice(string.ascii_letters + " ")
        pop.append(initialize_individual(genome, 0))
    return pop


def tree_len(root: Optional[Node]) -> int:
    """
    Purpose:        Get the length of a tree
    Parameters:     root: The root node of this tree or subtree
    User Input:     no
    Prints:         no
    Returns:        Length of the tree
    Modifies:       Nothing
    Calls:          Basic python
    """
    result = 0
    if root:
        result = 1 + tree_len(root.left) + tree_len(root.right)
    return result


def tree_get_nth(
    root: Optional[Node], n: int, current: list[int] = [-1]
) -> Optional[Node]:
    """
    Purpose:        Get a reference to the nth node of a tree (in the form of a 1-length list), following pre-ordering
    Parameters:     root: The root node of this tree or subtree
                    n: The index of the target node
                    current: Passing the maximum reached index through reference
    User Input:     no
    Prints:         no
    Returns:        The nth node of a tree, or none if n > tree_len(root)
    Modifies:       current
    Calls:          Basic python
    Complaints:     I miss pointers
    """

    if root:
        current[0] += 1  # Increase the value of current to match what's being inspected
        if current[0] == n:  # Return this node if it is the target
            return root
        else:  # If this is not the target, recurse this function on the left then right nodes
            temp = tree_get_nth(root.left, n, current)
            if current[0] < n:  # Calls to left node will increase current
                temp = tree_get_nth(root.right, n, current)
            return temp
    else:  # If this node does not exist, return None without increasing current
        return None


def tree_set_nth(root: Node, value: Node, index: int) -> Optional[Node]:
    """
    Purpose:        Get a reference to the nth node of a tree (in the form of a 1-length list), following pre-ordering
    Parameters:     root: The root node of this tree or subtree
                    n: The index of the target node
                    current: Passing the maximum reached index through reference
    User Input:     no
    Prints:         no
    Returns:        The nth node of a tree, or none if n > tree_len(root)
    Modifies:       current
    Calls:          Basic python
    Complaints:     I miss pointers
    """

    return _tree_set_nth(root, value, index, 0)[0]


def _tree_set_nth(
    root: Optional[Node], value: Node, index: int, count: int
) -> tuple[Optional[Node], int]:
    if root:
        if count == index:
            return (value, count)
        else:
            left, count = _tree_set_nth(root.left, value, index, count + 1)
            right, count = _tree_set_nth(root.right, value, index, count + 1)
            return Node(data=root.data, left=left, right=right), count
    else:
        return None, count - 1

def tree_deep_copy(root: Optional[Node]) -> Optional[Node]:
    if root:
        return Node(
            data=root.data,
            left=tree_deep_copy(root.left),
            right=tree_deep_copy(root.right),
        )
    else:
        return None

def tree_eval_node(root: Optional[Node], vars: list[float]) -> float:
    if root:
        match root.data:
            case '+':
                return np.add( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '-':
                return np.subtract( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '*':
                return np.multiply( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '/':
                return np.divide( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '^':
                if tree_eval_node(root.left, vars) < 0: # Negative to the power of a fraction results in a complevars number, which I don't want
                    test = -np.float_power(np.abs(tree_eval_node(root.left, vars)), tree_eval_node(root.right, vars))
                else:
                    test = np.float_power(tree_eval_node(root.left, vars), tree_eval_node(root.right, vars))
                if test == math.nan:
                    test = math.inf
                return test
                # return pow(tree_eval_node(root.left, vars), tree_eval_node(root.right, vars))
            case other:
                if root.data in VARIABLES:
                    return vars[VARIABLES.find(root.data)]
                else:
                    return float(root.data)
    pass

def recombine_pair(parent1: Individual, parent2: Individual) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          Basic python, random.choice-1, initialize_individual-2
    Example doctest:
    """

    genomeLen1 = tree_len(parent1["genome"])
    genomeLen2 = tree_len(parent2["genome"])

    index1 = random.randint(0, genomeLen1)
    index2 = random.randint(0, genomeLen2)

    subtree1 = tree_get_nth(parent1["genome"], index1)
    subtree2 = tree_get_nth(parent2["genome"], index2)

    if subtree1 and subtree2:
        genome1 = tree_set_nth(parent1["genome"], subtree2, index2)
        genome2 = tree_set_nth(parent2["genome"], subtree1, index1)

        if genome1 and genome2:
            return [
                initialize_individual(parse_tree_return(genome1), 0),
                initialize_individual(parse_tree_return(genome2), 0),
            ]
    return [
        initialize_individual(parse_tree_return(parent1["genome"]), 0),
        initialize_individual(parse_tree_return(parent2["genome"]), 0),
    ]


def recombine_group(parents: Population, recombine_rate: float) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          Basic python, random.random~n/2, recombine pair-n
    """
    pop = []
    weights = []
    for i in parents:
        weights.append(100.0 / i["fitness"])


    for i in range(int(len(parents))):  # [0 : len(parents) - 1 : 2]:

        if random.random() < recombine_rate:
            temp = recombine_pair(parents[i], random.choices(parents, weights=weights, k=1)[0])
            pop += temp
        else:
            pop.append(parents[i])
            # pop.append(parents[i + int(len(parents) / 2)])
        # for j in range(len(parents))[i + 1 :]:
        #    if random.random() < recombine_rate:
        #        temp = recombine_pair(parents[i], parents[j])
        #        pop += temp
        #    else:
        #        pop += parents[i]
        #        pop += parents[j]
    return pop


def major_mutate_individual(parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, random,choice-1,
    Example doctest:
    """
    # TODO: Add subtle mutation and change depth limit to current node depth + 1
    # as_string = parse_tree_return(parent["genome"])
    new_genome: Optional[Node] = tree_deep_copy(parent["genome"])
    while random.random() < mutate_rate:
        # new_genome = parse_expression(as_string)
        genome_len = tree_len(new_genome)
        n = random.randint(0, genome_len)
        # subtree = tree_get_nth(genome_copy, n)
        new_genome = tree_set_nth(
            new_genome, parse_expression(gen_rand_prefix_code(depth_limit=MUTATE_DEPTH, var_chance=0.25)), n
        )

        # if new_genome:
        #     as_string = parse_tree_return(new_genome)
        #     num_x = as_string.count("x")
        #     if num_x == 0:
        #         as_string = put_an_x_in_it(as_string)
        #     elif num_x > 1:
        #         preserve = random.randint(0, num_x)
        #         instances = 0
        #         as_arr = as_string.split()
        #         for i in range(num_x):
        #             if as_arr[i] == "x":
        #                 if instances != preserve:
        #                     as_arr[i] = gen_rand_prefix_code(2)
        #                 instances += 1
        #         as_string = "".join(as_arr)
        mutate_rate *= 0.8

        # print("Debug1 ----", parse_tree_return(parent["genome"]))
        # print("Debug2 ====", as_string)

    return {"genome": new_genome, "fitness": 0}

def minor_mutate_individual (parent: Individual, mutate_rate: float) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parents as Individual, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, random,choice-1,
    Example doctest:
    TODO: Change function descriptions to be accurate
    """
    # TODO: Add subtle mutation and change depth limit to current node depth + 1
    # as_string = parse_tree_return(parent["genome"])
    new_genome: Optional[Node] = tree_deep_copy(parent["genome"])
    genome_len = tree_len(new_genome)
    for i in range(genome_len):
        # new_genome = parse_expression(as_string)
        
        # subtree = tree_get_nth(genome_copy, n)

        temp_node = tree_get_nth(root=new_genome, n=i, current=[-1])
        if (random.random() < mutate_rate and temp_node.data not in OPERATORS and temp_node.data not in VARIABLES):
            temp_node.data = str(float(temp_node.data) + random.uniform(-mutate_rate, mutate_rate))

    return {"genome": new_genome, "fitness": 0}


def mutate_group(children: Population, minor_mutate_rate: float, major_mutate_rate: float) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          Basic python, mutate_individual-n
    Example doctest:
    """
    pop = []
    # i = 0
    for child in children:
        # i += 1
        pop.append(minor_mutate_individual( major_mutate_individual(child, major_mutate_rate), minor_mutate_rate ))  #  * (i / len(children))


    return pop



def evaluate_individual(individual: Individual, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
    Parameters:     One Individual, data formatted as IOdata
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Notes:          train/test format is like PSB2 (see IOdata above)
    Example doctest:
    >>> evaluate_individual(ind1, io_data)
    """
    fitness = 0
    errors = []
    base_eval_string = parse_tree_return(individual["genome"])
    for sub_eval in io_data:
        if 0:
            # TODO: Fix or delete this
            eval_string = base_eval_string.replace("x", str(sub_eval["input1"][0]))

            # In clojure, this is really slow with subprocess
            # eval_string = "( float " + eval_string + ")"
            # returnobject = subprocess.run(
            #     ["clojure", "-e", eval_string], capture_output=True
            # )
            # result = float(returnobject.stdout.decode().strip())

            # In python, this is MUCH MUCH faster:
            try:
                y = eval(prefix_to_infix(eval_string).replace("^", "**"))
            except OverflowError:
                y = math.inf
            except ZeroDivisionError:
                y = math.inf

            try:
                errors.append(abs(sub_eval["output1"] - y))
            except OverflowError:
                errors.append(math.inf)
        else:
            try:
                y = tree_eval_node(individual["genome"], sub_eval["input1"])
            except OverflowError:
                y = math.inf
            except ZeroDivisionError:
                y = math.inf
            except FloatingPointError:
                y = math.inf

            try:
                errors.append(abs(sub_eval["output1"] - y))
            except OverflowError:
                errors.append(math.inf)
    # Higher errors is bad, longer strings is bad, and more than 1 x is bad (For now)
    try:
        errors = sum(errors)
    except FloatingPointError:
        errors = math.inf

    if errors < ERROR_PRECISION:
        fitness = 1
    else:
        fitness = 1 + errors + tree_len(individual["genome"]) * GENOME_LEN_WEIGHT + ((base_eval_string.count("x")-1) ** 2) # + len(eval_string.split()) * 0.1
    # Higher fitness is worse
    individual["fitness"] = fitness





def evaluate_group(individuals: Population, io_data: IOdata) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          Basic python, evaluate_individual-n
    Example doctest:
    """
    for i in individuals:
        evaluate_individual(i, io_data)


def rank_group(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    for i in range(len(individuals) - 1):
        for j in range(len(individuals))[i + 1 :]:
            if individuals[i]["fitness"] > individuals[j]["fitness"]:
                temp = individuals[i]
                individuals[i] = individuals[j]
                individuals[j] = temp


def parent_select(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          Basic python, random.choices-1
    Example doctest:
    """
    weights = []
    for i in individuals:
        weights.append(100.0 / i["fitness"])

    return random.choices(individuals, weights=weights, k=number)


def survivor_select(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          Basic python only
    Example doctest:
    """
    return individuals[0:pop_size]


def evolve(io_data: IOdata, pop_size: int = DEFAULT_POP_SIZE) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The evolved population of solutions
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """
    # To debug doctest test in pudb
    # Highlight the line of code below below
    # Type 't' to jump 'to' it
    # Type 's' to 'step' deeper
    # Type 'n' to 'next' over
    # Type 'f' or 'r' to finish/return a function call and go back to caller
    population = initialize_pop(pop_size=pop_size)
    evaluate_group(individuals=population, io_data=io_data)
    rank_group(individuals=population)
    best_fitness = population[0]["fitness"]
    goal_fitness = 1
    time_without_improvement = 0
    i = 0

    print(
        "Iteration number",
        i,
        "with fitness",
        best_fitness,
        "and best individual =",
        parse_tree_return(population[0]["genome"]),
    )

    while best_fitness > goal_fitness and i < MAX_ITERATIONS and time_without_improvement < MAX_TIME_WITHOUT_IMPROVEMENT:
        # mutate_rate = math.log(best_fitness + 1 + time_without_improvement) / 2
        mutate_rate = (best_fitness) / (best_fitness + MUTATE_RATE_STEEPNESS)
        spontaneous = initialize_pop(pop_size=(int)(pop_size * SPONTANEOUS_PERCENTAGE), depth=4)
        evaluate_group(individuals=spontaneous, io_data=io_data)
        parents = parent_select(individuals=population, number=pop_size) + spontaneous
        rank_group(parents)
        children = recombine_group(parents=parents, recombine_rate=mutate_rate / RECOMBINE_RATE_MULTIPLIER)
        mutants = mutate_group(children=children, minor_mutate_rate=mutate_rate, major_mutate_rate=mutate_rate)
        evaluate_group(individuals=mutants, io_data=io_data)
        everyone = population + mutants
        rank_group(individuals=everyone)

        # print("\t\tDEBUG DEBUG", everyone[0]["fitness"], parse_tree_return(population[0]["genome"]))
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        i += 1

        if best_fitness != population[0]["fitness"]:
            best_fitness = population[0]["fitness"]
            print(
                "\nIteration number",
                i,
                "with fitness",
                best_fitness,
                "and best individual =",
                parse_tree_return(population[0]["genome"]),
            )
            time_without_improvement = 0
        else:
            time_without_improvement += 1

    print(
        "Completed with",
        i,
        "iterations (",
        time_without_improvement,
        "without improvement ) and a final fitness of",
        population[0]["fitness"],
    )
    return population


# Seed for base grade.
# For the exploratory competition points (last 10),
# comment this one line out if you want, but put it back please.
seed = True


if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)
    np.seterr(all="raise")
    for i in range(10):

        if seed:
            random.seed(420+i)

        print(divider)
        print("Cycle number: ", i + 1)
        print("Lower fitness is better.")
        print(divider)

        print("Equation to be found:")
        
        varS = random.randint(1, 14)
        varG = random.randint(1, 7)
        varT = random.randint(-5, 1)
        varF = random.randint(0, 5)

        X = list(range(-10, 110, 10))
        repeat = True
        while repeat:
            genome = gen_rand_prefix_code(depth_limit=4)
            genome = populate_with_variables(genome)
            ind1 = initialize_individual(genome, 0)
            repeat = False
            try:
                Y = [tree_eval_node(ind1["genome"], [x, varS, varG, varT, varF]) for x in X]
            except OverflowError:
                repeat = True
            except ZeroDivisionError:
                repeat = True
            except FloatingPointError:
                y = math.inf
        # data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]
        # mypy wanted this:
        data = [initialize_data(input1=[x, varS, varG, varT, varF], output1=y) for x, y in zip(X, Y)]

        # Correct:
        
        evaluate_individual(ind1, data)
        print_tree(ind1["genome"])
        print("Fitness", ind1["fitness"])

        # Yours
        train = data[: int(len(data) / 2)]
        test = data[int(len(data) / 2) :]
        population = evolve(train)
        evaluate_individual(population[0], test)
        population[0]["fitness"]

        print("Here is the best program:")
        parse_tree_print(population[0]["genome"])
        print("And it's fitness:")
        print(population[0]["fitness"])

