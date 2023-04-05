#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
A simple randomized unit test.
DO NOT EDIT THIS FILE!
"""

import random
from gp_lisp import *


scores = []
counter = 0
while counter < 10:
    while True:
        print("======================================= Test", counter)
        rand_code = gen_rand_prefix_code(depth_limit=4)
        rand_code = populate_with_variables(rand_code)
        print("Prefix code:")
        parse_tree_print(parse_expression(rand_code))
        test_code = prefix_to_infix(rand_code)
        print(f"\nInfix code:\n{test_code}")
        X = random.choices(range(-100, 100), k=10)
        try:
            Y = [eval(test_code) for x in X]
            counter += 1
            break
        except ZeroDivisionError:
            pass

    data = [{"input1": x, "output1": y} for x, y in zip(X, Y)]

    # Yours:
    train = data[: int(len(data) / 2)]
    test = data[int(len(data) / 2) :]
    population = evolve(train)
    evaluate_individual(population[0], test)
    scores.append(population[0]["fitness"])

scores.sort()
# mean_score = sum(scores) / len(scores)
mean_score = sum(scores[:5]) / 5
grade = 100 - mean_score
print(f"Your grade is {grade}")
with open("results.txt", mode="w") as fhand:
    fhand.write(str(round(grade)))
with open("mean_score.txt", mode="w") as fhand:
    fhand.write(str(mean_score))
