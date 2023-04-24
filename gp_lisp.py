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
from typing import TypedDict
from typing import Optional

import random
import math
import numpy
import sympy

import requests
import json
import time
import multiprocessing
import os
import ctypes

import pickle
# from multiprocessing import Process, Manager

# import datetime
# import subprocess


# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
OPERATORS = "+-/*"
VARIABLES = "xsgtfz"
THROW_TYPES = ["bh1", "bh2", "bh3"] #["bh1", "bh2", "bh3", "fh1", "fh2", "fh3"]
EXTRA_INCLUDE_DISCS = ["566", "901", "348", "1473", "1228", "1418", "569"]
THROW_VALUES = [0.5, 1, 2]#[0.5, 1, 2, -0.5, -1, -2]
STARTING_POINTS = []
# STARTING_POINTS = ["( / ( + ( * 0.00000008 ( * ( * x x ) ( * x x ) ) ) ( + ( * -0.00005 ( * ( * x x ) x ) ( + ( * 0.0022 ( * x x ) ) ( + ( * -0.1159 x ) 0.157 ) ) ) ) s )",
#                    "( / ( + ( * 0.00000008 ( * ( * x x ) ( * x x ) ) ) ( + ( * -0.00005 ( * ( * x x ) x ) ( + ( * 0.0022 ( * x x ) ) ( + ( * -0.1159 x ) 0.157 ) ) ) ) ( * g s ) )",
#                    "( * ( + x -13.33020662 ) ( / ( - t -0.6916947107793106 ) ( + 1.60724945 g ) ) )",
#                    "( * ( / x ( / 10.527460278031175 s ) ) ( / ( + t ( / ( / x ( - s -16.20267838447866 ) ) 12.473110664082899 ) ) ( - s ( / x ( - ( / g -13.612437828992574 ) ( + f ( * -15.587341762831622 5.898199825906487 ) ) ) ) ) ) )",
#                    "( / ( - ( * x ( + f t ) ) ( / x ( - 5.22950121933327 ( + ( + ( / 0.5424341939243978 -20.36314492693719 ) f ) s ) ) ) ) ( + 7.799210582665918 f ) )",
#                    "( + ( / ( ^ g ( / ( / ( - x -16.758861668806883 ) 20.590543371048533 ) g ) ) ( - s -1.5583939877949127 ) ) ( * ( / ( + f x ) ( - f -17.483687805468257 ) ) t ) )",
#                    "( + ( / ( ^ g ( / ( / ( - x -18.585200573306228 ) 20.508385901494847 ) g ) ) ( - s -1.8761568445068173 ) ) ( * ( / ( + f x ) ( - f ( - -4.269580602686558 f ) ) ) t ) )",
#                    "( / ( - x ( * g 8.119253988229453 ) ) ( + ( + ( - 10.199344301353479 ( / x 15.238381846367316 ) ) g ) ( - s ( ^ f t ) ) ) )",
#                    "( / ( + ( - ( * t x ) ( / ( * ( + x s ) s ) 17.041905716120535 ) ) ( ^ f ( / x ( * 8.04166262142913 g ) ) ) ) ( * f ( + f g ) ) )",
#                    "( / ( - ( * x ( + t t ) ) ( / x g ) ) ( + 6.990199656711925 ( + s f ) ) )"
#                    ]

NUM_RAND_DISCS = 16
MAX_ITERATIONS = 3000
MAX_TIME_WITHOUT_IMPROVEMENT = 600
DEFAULT_POP_SIZE = 500
PARENT_PERCENTAGE = 1
DEFAULT_DEPTH = 4
POLYNOMIAL_DEGREE = 4
MAJOR_MUTATE_RATE_STEEPNESS = 8.0
MINOR_MUTATE_RATE_STEEPNESS = 20.0
RECOMBINE_RATE_STEEPNESS = 8.0
MUTATE_EARLY_STOP_CHANCE = 0.5
MUTATE_VAR_CHANCE = 0.5
MUTATE_AGAIN_MULTIPLIER = 0.85
SIMPLIFY_CHANCE = 0
SPONTANEOUS_PERCENTAGE = 0.1
SPONTANEOUS_DEPTH = 5
ERROR_PRECISION = 0.01
GENOME_LEN_WEIGHT = 0.01
GENOME_LEN_HARD_LIMIT = 500
MISSING_VAR_COST = 50
ROUNDING_DECIMALS = 5
NUM_PROCS = 4 # os.cpu_count()

USE_SEED = False
USE_PYTHON_EVAL = True # Python's EVAL() function tends to be faster, but is more complex and can freeze sometimes on exponentiation
USE_MULTIPROCESSING_FOR_EVAL = True
USE_MULTIPROCESSING_FOR_MUTATE = False


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
        self, data: str, left: Optional["Node"] = None, right: Optional["Node"] = None, mutable: bool = True
    ) -> None:
        self.data = data
        self.left = left
        self.right = right
        self.mutable = mutable


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
    if code[0] in OPERATORS + "^":
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


def parse_tree_return_string(root: Node) -> str:
    """
    Stringifies to the tree data structure, returns string.
    >>> stringified = parse_tree_return_string(root)
    """
    if root.right is not None and root.left is not None:
        return f"( {root.data} {parse_tree_return_string(root.left)} {parse_tree_return_string(root.right)} )"
    else:
        # for the case of literal programs... e.g., `4`
        return root.data
    

def initialize_individual(genome: str, fitness: float = 0) -> Individual:
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

def initialize_individual(genome: Node, fitness: float = 0) -> Individual:
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
    return {"genome": genome, "fitness": fitness}


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
        if prefix_arr[i] not in OPERATORS + "^":
            stack.append(prefix_arr[i])
            i -= 1
        else:
            str = "(" + stack.pop() + prefix_arr[i] + stack.pop() + ")"
            stack.append(str)
            i -= 1
    return stack.pop()


def populate_with_variables(formula: str, vars: str = VARIABLES) -> str:
    formula_arr = formula.split()
    for var in vars:
        count = 0
        if var not in formula_arr:
            while var == VARIABLES[0] and count < 50:
                i = random.randint(0, len(formula_arr) - 1)
                if formula_arr[i] not in OPERATORS and formula_arr[i] not in vars:
                    formula_arr[i] = var
                    break
                count += 1
    return " ".join(formula_arr)


def gen_rand_prefix_code(depth_limit: int, rec_depth: int = 0, stop_early_chance: float = 0.2, var_chance: float = 0) -> str:
    """
    Generates one small formula,
    from OPERATORS and ints from -100 to 200
    """
    rec_depth += 1
    if rec_depth < depth_limit:
        if random.random() >= stop_early_chance:
            return (
                random.choice(OPERATORS)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth, stop_early_chance, var_chance)
                + " "
                + gen_rand_prefix_code(depth_limit, rec_depth, stop_early_chance, var_chance)
            )
        elif random.random() < var_chance:
            return random.choice(VARIABLES)
        else:
            return str(numpy.round(random.uniform(-20.0, 20.0), ROUNDING_DECIMALS))
    elif random.random() < var_chance:
        return random.choice(VARIABLES)
    else:
        return str(numpy.round(random.uniform(-20.0, 20.0), ROUNDING_DECIMALS))


def gen_rand_polynomial(degree: int, depth_limit: int, rec_depth: int = 0, stop_early_chance: float = 0.2, var_chance: float = 0) -> Node:
    if degree > 0:
        root: Node = Node (
            data="+",
            left=Node(
                data="*",
                left=parse_expression(gen_rand_prefix_code(depth_limit=3, stop_early_chance=MUTATE_EARLY_STOP_CHANCE, var_chance=MUTATE_VAR_CHANCE)),
                right=Node(
                    data="^",
                    left=parse_expression(populate_with_variables(gen_rand_prefix_code(depth_limit=3, stop_early_chance=MUTATE_EARLY_STOP_CHANCE, var_chance=MUTATE_VAR_CHANCE), VARIABLES[0])),
                    right=Node(
                        data=str(degree),
                        left=None,
                        right=None,
                        mutable=False
                    ),
                    mutable=False
                ),
                mutable=False
            ), # if degree > 1 else parse_expression(populate_with_variables(gen_rand_prefix_code(depth_limit=3, stop_early_chance=MUTATE_EARLY_STOP_CHANCE, var_chance=MUTATE_VAR_CHANCE), VARIABLES[0])),
            right=gen_rand_polynomial(degree=degree-1, depth_limit=depth_limit, rec_depth=rec_depth, stop_early_chance=stop_early_chance, var_chance=var_chance),
            mutable=False
        )
    else:
        root: Node = parse_expression(gen_rand_prefix_code(depth_limit=3, stop_early_chance=MUTATE_EARLY_STOP_CHANCE, var_chance=MUTATE_VAR_CHANCE))
    return root


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
        # genome = gen_rand_prefix_code(depth_limit=DEFAULT_DEPTH)
        # genome = populate_with_variables(genome)

        # for j in range(len(example_genome)):
        #    genome += random.choice(string.ascii_letters + " ")

        genome = gen_rand_polynomial(degree=POLYNOMIAL_DEGREE, depth_limit=depth)
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

def tree_depth(root: Optional[Node]) -> int:
    if root:
        left = tree_depth(root.left)
        right = tree_depth(root.right)
        if left < right:
            return right + 1
        else:
            return left + 1
    else:
        return 0
    pass

def tree_get_nth(
    root: Optional[Node], n: int
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

    temp_node, _ = _tree_get_nth(root, n, 0)
    return temp_node

def _tree_get_nth(root: Optional[Node], n: int, current: int) -> tuple[Node, int]:
    if root:
        if current == n:
            return root, current
        temp_node, temp_num = _tree_get_nth(root.left, n, current + 1)
        current = temp_num
        if current == n:
            return temp_node, current
        temp_node, temp_num = _tree_get_nth(root.right, n, current + 1)
        current = temp_num
        return temp_node, current
    else:
        return None, current - 1

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
    assert type(root) == Node
    return _tree_set_nth(root, value, index, 0)[0]


def _tree_set_nth(
    root: Optional[Node], value: Node, index: int, count: int
) -> tuple[Optional[Node], int]:
    assert (type(root) == Node or root == None)
    if root:
        if count == index:
            return (value, count)
        else:
            left, count = _tree_set_nth(root.left, value, index, count + 1)
            right, count = _tree_set_nth(root.right, value, index, count + 1)
            return Node(data=root.data, left=left, right=right, mutable=root.mutable), count
    else:
        return None, count - 1

def tree_deep_copy(root: Optional[Node]) -> Optional[Node]:
    if root:
        return Node(
            data=root.data,
            left=tree_deep_copy(root.left),
            right=tree_deep_copy(root.right),
            mutable=root.mutable
        )
    else:
        return None

def tree_eval_node(root: Optional[Node], vars: list[float]) -> float:
    if root:
        match root.data:
            case '+':
                return numpy.add( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '-':
                return numpy.subtract( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '*':
                return numpy.multiply( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '/':
                return numpy.divide( tree_eval_node(root.left, vars), tree_eval_node(root.right, vars) )
            case '^':
                test = numpy.float_power(tree_eval_node(root.left, vars), tree_eval_node(root.right, vars))
                if test == math.nan:
                    test = math.inf
                return test
                # return pow(tree_eval_node(root.left, vars), tree_eval_node(root.right, vars))
            case other:
                if str(root.data) in VARIABLES:
                    return vars[VARIABLES.find(root.data)]
                else:
                    return float(root.data)
    pass

def tree_attempt_simplify(root: Optional[Node]) -> str:
    if root:
        if str(root.data) in VARIABLES:
            return "var"
        elif str(root.data) in OPERATORS:
            left = tree_attempt_simplify(root.left)
            right = tree_attempt_simplify(root.right)
            if left == "num" and right == "num":
                try:
                    root.data = str(tree_eval_node(root, [])) # I can only do this because I know the function won't attempt to use the vars parameter
                except FloatingPointError:
                    pass
                root.left = None
                root.right = None
                return "num"
            else: # TODO: Simplify more cases too
                return "opr"
        else:
            return "num"        
    else:
        return "nul"

def tree_polynomial_get_term(root: Optional[Node], term: int):
    pass

def recombine_pair(parent1: Individual, parent2: Individual, rate: float) -> Population:
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

    # This new (uncommment) recombination code will swap the nth degree term of on parent with the nth degree term of another parent
    # This part relies on the genomes being very specifically formatted polynomials of equal length
    # If I add functionality for increasing the degree of the polynomial with mutation, this would need to change
    # TODO: Make this code safer
    # TODO: Maybe swap the entire term at once, not just one part?

    genome1: Node = tree_deep_copy(parent1["genome"])
    genome2: Node = tree_deep_copy(parent2["genome"])

    current1: Node = genome1
    current2: Node = genome2
    
    for _ in range(POLYNOMIAL_DEGREE):
        # x^(POLYNOMIAL_DEGREE-i)'th term
        if random.random() < rate:
            subtree1: Node = current1.left
            subtree2: Node = current2.left

            copy: Node = Node(
                data = subtree1.data,
                left = subtree1.left,
                right = subtree1.right,
                mutable = subtree1.mutable
            )

            subtree1.data = subtree2.data
            subtree1.left = subtree2.left
            subtree1.right = subtree2.right
            subtree1.mutable = subtree2.mutable

            subtree2.data = copy.data
            subtree2.left = copy.left
            subtree2.right = copy.right
            subtree2.mutable = copy.mutable

        current1 = current1.right
        current2 = current2.right

    # x^0'th term
    if random.random() < rate:
        copy: Node = Node(
            data = current1.data,
            left = current1.left,
            right = current1.right,
            mutable = current1.mutable
        )

        current1.data = current2.data
        current1.left = current2.left
        current1.right = current2.right
        current1.mutable = current2.mutable

        current2.data = copy.data
        current2.left = copy.left
        current2.right = copy.right
        current2.mutable = copy.mutable

    return [
        initialize_individual(genome1, 0),
        initialize_individual(genome2, 0)
    ]

    # genomeLen1 = tree_len(parent1["genome"])
    # genomeLen2 = tree_len(parent2["genome"])

    # while True:

    #     index1 = random.randint(0, genomeLen1 - 1)
    #     index2 = random.randint(0, genomeLen2 - 1)

    #     subtree1 = tree_get_nth(root=parent1["genome"], n=index1)
    #     subtree2 = tree_get_nth(root=parent2["genome"], n=index2)

    #     assert(subtree1)
    #     assert(subtree2)
    #     if (subtree1.mutable and subtree2.mutable):

    #         genome1 = tree_set_nth(parent1["genome"], subtree2, index2)
    #         genome2 = tree_set_nth(parent2["genome"], subtree1, index1)

    #         assert (type(genome1) == Node or genome1 == None)
    #         assert (type(genome2) == Node or genome2 == None)

    #         if genome1 and genome2:
    #             return [
    #                 initialize_individual(parse_tree_return_string(genome1), 0),
    #                 initialize_individual(parse_tree_return_string(genome2), 0),
    #             ]
    #         return [
    #             initialize_individual(parse_tree_return_string(parent1["genome"]), 0),
    #             initialize_individual(parse_tree_return_string(parent2["genome"]), 0),
    #         ]


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

        # if random.random() < recombine_rate:
        temp = recombine_pair(parents[i], random.choices(parents, weights=weights, k=1)[0], recombine_rate)
        pop += temp
        # else:
        #     pop.append(parents[i])
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
    # TODO: USe similar code to recombine_pair
    # as_string = parse_tree_return_string(parent["genome"])
    new_genome: Optional[Node] = tree_deep_copy(parent["genome"])
    while random.random() < mutate_rate:
        while True:
            # new_genome = parse_expression(as_string)
            genome_len = tree_len(new_genome)

            if genome_len > GENOME_LEN_HARD_LIMIT:
                return {"genome": new_genome, "fitness": math.inf}
            else:
                n = random.randint(0, genome_len - 1)

                if random.random() < SIMPLIFY_CHANCE:
                    tree_attempt_simplify(tree_get_nth(root=new_genome, n=n))
                else:
                    subtree = tree_get_nth(root=new_genome, n=n)
                    assert(subtree)
                    if subtree.mutable:
                        new_genome = tree_set_nth(
                            new_genome, parse_expression(gen_rand_prefix_code(depth_limit=2, stop_early_chance=MUTATE_EARLY_STOP_CHANCE, var_chance=MUTATE_VAR_CHANCE)), n
                        )
                        mutate_rate *= MUTATE_AGAIN_MULTIPLIER
                        break

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
    new_genome: Optional[Node] = tree_deep_copy(parent["genome"])
    genome_len = tree_len(new_genome)
    
    if genome_len > GENOME_LEN_HARD_LIMIT:
        return {"genome": new_genome, "fitness": math.inf}
    else:
        for i in range(genome_len):
            temp_node = tree_get_nth(root=new_genome, n=i)
            assert(temp_node)
            if temp_node.mutable:
                node_data = str(temp_node.data)
                if (node_data in OPERATORS):
                    if (random.random() < mutate_rate / 4.0):
                        temp_node.data = random.choice(OPERATORS)
                elif (node_data in VARIABLES):
                    if (random.random() < mutate_rate / 4.0):
                        temp_node.data = random.choice(VARIABLES)
                elif (random.random() < mutate_rate):
                    temp_node.data = str(numpy.round(float(temp_node.data) + random.uniform(-mutate_rate, mutate_rate) * 10.0, ROUNDING_DECIMALS))

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

    if USE_MULTIPROCESSING_FOR_MUTATE:
        pop = [None] * len(children)
        
        # i = 0

        procs = [multiprocessing.Process(target=_mutate, args=(pop, children[i], i, minor_mutate_rate, major_mutate_rate,)) for i in range(len(children))]
        for t in procs:
            t.start()
        for t in procs:
            t.join()

    else:
        pop = []
        for child in children:
            # i += 1
            pop.append(minor_mutate_individual( major_mutate_individual(child, major_mutate_rate), minor_mutate_rate ))  #  * (i / len(children))


    return pop

def _mutate (pop: Population, child: Individual, index: int, minor_mutate_rate: float, major_mutate_rate: float) -> None:
    pop[index] = minor_mutate_individual( major_mutate_individual(child, major_mutate_rate), minor_mutate_rate )

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
    if individual["fitness"] == math.inf:
        return # This should only happen if the tree has a length above the hard cutoff
    numpy.seterr(all="raise")
    fitness = 0

    # Enforce that this MUST be a polynomial
    current: Node = individual["genome"]
    for _ in range(POLYNOMIAL_DEGREE):
        # x^(POLYNOMIAL_DEGREE-i)'th term
        subtree: Node = current.left

        subtree_arr = parse_tree_return_string(subtree.right.left).split()
        x_count = 0

        # Don't let it have more than one x, otherwise it'll do x-x
        for c in range(len(subtree_arr)):
            if subtree_arr[c] == VARIABLES[0]:
                x_count += 1
                if x_count > 1:
                    subtree_arr[c] = str(numpy.round(random.uniform(-20.0, 20.0), ROUNDING_DECIMALS))
                    break

        if x_count != 1:
            subtree_str = "".join(subtree_arr)
            subtree.right.left = parse_expression(populate_with_variables(subtree_str, VARIABLES[0]))
        # if "x" not in subtree_str:
            # individual["fitness"] = math.inf
            # return
        current = current.right
    
    # Enforce maximum length restriction
    length = tree_len(individual["genome"])
    if length > GENOME_LEN_HARD_LIMIT:
        fitness = math.inf
        return

    errors: list[float] = []
    eval_string = parse_tree_return_string(individual["genome"])
    for sub_eval in io_data:
        try:
            # TODO: Support trig functions

            if USE_PYTHON_EVAL:
                as_list = eval_string.split()
                for v in range(len(VARIABLES)):
                    for x in range(len(as_list)):
                        if as_list[x] == VARIABLES[v]:
                            as_list[x] = str(sub_eval["input1"][v])
                new_eval_string = prefix_to_infix(" ".join(as_list)).replace("^", "**")
                y = eval(new_eval_string)
                if numpy.iscomplexobj(y) or numpy.isnan(float(y)):
                    errors = [math.inf]
                    break

            else:
                y = tree_eval_node(individual["genome"], sub_eval["input1"])
            errors.append(abs(sub_eval["output1"] - y))
        except OverflowError:
            errors = [math.inf]
            break
        except ZeroDivisionError:
            errors = [math.inf]
            break
        except FloatingPointError:
            errors = [math.inf]
            break

    # Higher errors is bad, longer strings is bad
    error_sum: float
    try:
        error_sum = numpy.sum(errors) / len(io_data)
    except FloatingPointError:
        error_sum = math.inf

    if error_sum < ERROR_PRECISION:
        fitness = 1
    else:
        # Tally results and punish longer genomes
        fitness = 1 + error_sum + length * GENOME_LEN_WEIGHT # + len(eval_string.split()) * 0.1

        # Punish the genome for not including variables
        for v in VARIABLES[1::]:
            if v not in eval_string:
                fitness += MISSING_VAR_COST

    # Higher fitness is worse
    individual["fitness"] = fitness


def evaluate_group(individuals: Population, io_data: IOdata) -> Population:
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

    if USE_MULTIPROCESSING_FOR_EVAL:
        procs = []
        ilen = len(individuals)
        out_queue = multiprocessing.Queue()
        for x in range(NUM_PROCS):
            procs.append (multiprocessing.Process(target=_eval_group, args=(individuals[math.floor( ilen / NUM_PROCS * x ) : math.floor( ilen / NUM_PROCS * (x+1) )], io_data, out_queue)))
        for t in procs:
            t.start()
        individuals = []
        for t in range(ilen):
            while True:
                try:
                    individuals.append(out_queue.get(timeout=0.01))
                    # print("Adding " + str(t) + " with fitness " + str(individuals[-1]["fitness"]))
                    break
                except Exception as e:
                    pass
        # print("Exiting while loop")
        for t in procs:
            t.join()    
        
    else:
        for i in individuals:
            evaluate_individual(i, io_data)

    return individuals

def _eval_group(individuals: Population, io_data: IOdata, queue: multiprocessing.Queue):
    for i in range(len(individuals)) :
        evaluate_individual(individuals[i], io_data)
        while queue.qsize() > len(individuals):
            pass
        # print("Inserting into queue")
        queue.put(individuals[i])
    # queue.put(individuals)
    queue.close()
    return

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

    result = []

    for i in range(int(number)):
        result.append(random.choices(individuals, weights=weights, k=1)[0])

    # return random.choices(individuals, weights=weights, k=number)
    return result


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


def evolve(io_data: IOdata, pop_size: int = DEFAULT_POP_SIZE, saved_genomes: Node = []) -> Population:
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
    for starting_point in STARTING_POINTS:
        population = [initialize_individual(starting_point, 0)] + population
    for saved in saved_genomes:
        population = [initialize_individual(saved, 0)] + population
    population = evaluate_group(individuals=population, io_data=io_data)
    rank_group(individuals=population)
    best_fitness = population[0]["fitness"]
    goal_fitness = 1
    time_without_improvement = 0
    i = 0
    start: float
    end: float
    time_data = []

    print(
        "Iteration number",
        i,
        "with fitness",
        best_fitness,
        "and best individual =",
        parse_tree_return_string(population[0]["genome"]),
    )

    while best_fitness > goal_fitness and i < MAX_ITERATIONS and time_without_improvement < MAX_TIME_WITHOUT_IMPROVEMENT:
        # mutate_rate = math.log(best_fitness + 1 + time_without_improvement) / 2
        time_data = []
        start = time.time()
        major_mutate_rate = (best_fitness) / (best_fitness + (MAJOR_MUTATE_RATE_STEEPNESS))
        minor_mutate_rate = (best_fitness) / (best_fitness + (MINOR_MUTATE_RATE_STEEPNESS))
        recombine_rate = ((best_fitness) / (best_fitness + (RECOMBINE_RATE_STEEPNESS))) / 2
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        spontaneous = initialize_pop(pop_size=(int)(pop_size * SPONTANEOUS_PERCENTAGE), depth=SPONTANEOUS_DEPTH)
        spontaneous = evaluate_group(individuals=spontaneous, io_data=io_data)
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        parents = parent_select(individuals=population, number=pop_size*PARENT_PERCENTAGE)
        rank_group(parents)
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        children = recombine_group(parents=parents, recombine_rate=recombine_rate)
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        mutants = mutate_group(children=children, minor_mutate_rate=minor_mutate_rate, major_mutate_rate=major_mutate_rate)
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        mutants = evaluate_group(individuals=mutants, io_data=io_data)
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        everyone = population + mutants + spontaneous
        rank_group(individuals=everyone)
        end = time.time()
        time_data.append(end - start)

        start = time.time()
        # print("\t\tDEBUG DEBUG", everyone[0]["fitness"], parse_tree_return_string(population[0]["genome"]))
        population = survivor_select(individuals=everyone, pop_size=pop_size)
        end = time.time()
        time_data.append(end - start)
        i += 1

        if best_fitness != population[0]["fitness"]:
            best_fitness = population[0]["fitness"]
            print(
                "\nIteration number",
                i,
                "with fitness",
                best_fitness,
                "and best individual =",
                parse_tree_return_string(population[0]["genome"]),
            )
            file = open("results.txt", "a")
            file.writelines(
                "\nIteration number " +
                str(i) +
                " with fitness " +
                str(best_fitness) +
                " and best individual = " +
                parse_tree_return_string(population[0]["genome"]) + 
                "\n\t>Infix = " + prefix_to_infix(parse_tree_return_string(population[0]["genome"])) + "\n"
            )
            file.close()
            if len(saved_genomes) > 0:
                saved_genomes[-1] = (tree_deep_copy(population[0]["genome"]))
            else:
                saved_genomes.append(tree_deep_copy(population[0]["genome"]))
            with open("polynomial-results-" + str(POLYNOMIAL_DEGREE) + ".pkl", "wb") as output:
                for node in saved_genomes:
                    pickle.dump(node, output, pickle.HIGHEST_PROTOCOL)
            time_without_improvement = 0
        else:
            time_without_improvement += 1
        print("\t>>Iteration ", i, "timing data: ", time_data)

    print(
        "Completed with",
        i,
        "iterations (",
        time_without_improvement,
        "without improvement ) and a final fitness of",
        population[0]["fitness"],
    )
    file = open("results.txt", "a")
    file.writelines(
        "Completed with " +
        str(i) +
        " iterations (" +
        str(time_without_improvement) +
        " without improvement) and a final fitness of " +
        str(population[0]["fitness"])
    )
    file.close()
    saved_genomes.append(tree_deep_copy(population[0]["genome"]))
    # TODO: Find out how to append to pickle?
    with open("polynomial-results-" + str(POLYNOMIAL_DEGREE) + ".pkl", "wb") as output:
        for node in saved_genomes:
            pickle.dump(node, output, pickle.HIGHEST_PROTOCOL)
    return population

def download_disc(disc: str, flipflop: bool) -> tuple[IOdata, bool]:
    response = requests.get('https://flightcharts.dgputtheads.com/discdata/' + disc)
    print("Importing data for disc ", disc)
    data = json.loads(response.text)["data"]
    result: list[IOdata] = []
    if response.ok and data != '""':
        data = json.loads(data)
        speed = data["speed"]
        glide = data["glide"]
        turn = data["turn"]
        fade = data["fade"]
        flipflop = not flipflop
        
        for i in range(len(THROW_TYPES)):
            for point in data[THROW_TYPES[i]][flipflop::2]:
                result.append(initialize_data([point["y"], speed, glide, turn, fade, THROW_VALUES[i]], point["x"] * 100.0)) # I multiply x by 100 to make it clearer. It is a percentage
                print("\t> ", result[-1])
        return result, flipflop
    else:
        print("** Failed to import data for disc ", disc, " **")
        raise Exception("Failed to import disc")
    

if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    saved_genomes = []
    flipflop = 0
    with open("polynomial-results-" + str(POLYNOMIAL_DEGREE) + ".pkl", "rb") as input:
        while True:
            try:
                temp = pickle.load(input)
                print("Importing node from file: ", parse_tree_return_string(temp))
                saved_genomes.append(temp)
            except EOFError:
                print("End of file")
                break

    
    print("\n\n" + divider +
        "\nStarting a new run: Polynomial of degree " + str(POLYNOMIAL_DEGREE) +
        "\nLower fitness is better:\n\n")
    file = open("results.txt", "a")
    file.writelines(
        divider +
        "\nStarting a new run: Polynomial of degree " + str(POLYNOMIAL_DEGREE) +
        "\nLower fitness is better:\n\n"
    )
    file.close()
    if USE_SEED:
        random.seed(42)

    all_data = []

    # Import random discs
    for x in range(NUM_RAND_DISCS):
        while True:
            try:
                disc = str(random.randint(1, 1600))
                temp, flipflop = download_disc(disc, flipflop)
                all_data += temp
                break
            except Exception as e:
                if e.args[0] == 'Failed to import disc':
                    pass
                else:
                    raise e
        
    # Import specific discs
    for disc in EXTRA_INCLUDE_DISCS:
        temp, flipflop = download_disc(disc, flipflop)
        all_data += temp
            

    # train = all_data[: int(len(all_data) / 2)]
    # test = all_data[int(len(all_data) / 2) :]
    file.close()
    population = evolve(all_data, DEFAULT_POP_SIZE, saved_genomes)
    evaluate_individual(population[0], all_data)
    population[0]["fitness"]

    print(divider)

    print("Here is the best program:")
    parse_tree_print(population[0]["genome"])
    print("And as infix:")
    print(prefix_to_infix(parse_tree_return_string(population[0]["genome"])))
    print("And it's fitness:")
    print(population[0]["fitness"])


# %%
