"""A small synthetic prompt -> code dataset for smoke tests (nine templates)."""
import json
import random

FUNCS = ["add", "total", "compute", "calc", "combine", "process", "handle", "solve", "run", "check",
         "measure", "score", "apply", "build", "update", "reduce", "scale", "pick", "fold", "mix"]
VARS = ["a", "b", "x", "y", "n", "m", "val", "num", "left", "right", "first", "second", "p", "q"]
LISTS = ["xs", "items", "values", "nums", "data", "arr", "scores", "prices"]
CLASSES = ["Point", "Counter", "Box", "Node", "Account", "Timer", "Cache", "Queue"]
ATTRS = ["count", "size", "name", "value", "total", "limit", "speed", "level"]
OPS = {"sum": "+", "difference": "-", "product": "*", "quotient": "/"}


def toy_example(r):
    f, (a, b), xs, d = r.choice(FUNCS), r.sample(VARS, 2), r.choice(LISTS), str(r.randrange(10))
    C, attr, v = r.choice(CLASSES), r.choice(ATTRS), r.choice(["i", "j", "item", "e"])
    k = r.randrange(9)
    if k == 0:
        op = r.choice(list(OPS))
        p = r.choice(["write a function {f} that returns the {op} of {a} and {b}",
                      "define {f} taking {a} and {b} and return their {op}"]).format(f=f, op=op, a=a, b=b)
        return p, f"def {f}({a}, {b}):\n    return {a} {OPS[op]} {b}"
    if k == 1:
        p = f"write a function {f} that returns {a} if it is positive and {d} otherwise"
        return p, f"def {f}({a}):\n    if {a} > 0:\n        return {a}\n    else:\n        return {d}"
    if k == 2:
        p = r.choice(["write {f} that returns the larger of {a} and {b}",
                      "define {f} returning the maximum of {a} and {b}"]).format(f=f, a=a, b=b)
        return p, f"def {f}({a}, {b}):\n    if {a} > {b}:\n        return {a}\n    return {b}"
    if k == 3:
        p = f"write a function {f} that adds up all elements of {xs} using {v}"
        return p, f"def {f}({xs}):\n    total = 0\n    for {v} in {xs}:\n        total += {v}\n    return total"
    if k == 4:
        p = f"write {f} that counts the elements of {xs} greater than {d}"
        return p, (f"def {f}({xs}):\n    count = 0\n    for {v} in {xs}:\n        if {v} > {d}:\n"
                   f"            count += 1\n    return count")
    if k == 5:
        p = f"write {f} that prints {a} {d} times"
        return p, f"def {f}({a}):\n    for i in range({d}):\n        print({a})"
    if k == 6:
        p = r.choice(["create a class {C} whose {attr} starts at {d}",
                      "define class {C} with attribute {attr} initialized to {d}"]).format(C=C, attr=attr, d=d)
        return p, f"class {C}:\n    def __init__(self):\n        self.{attr} = {d}"
    if k == 7:
        p = f"write {f} that halves {a} while it is at least {d} and returns it"
        return p, f"def {f}({a}):\n    while {a} >= {d}:\n        {a} = {a} / 2\n    return {a}"
    p = f"write {f} that returns {a} divided by {b} or {d} on division by zero"
    return p, f"def {f}({a}, {b}):\n    try:\n        return {a} / {b}\n    except ZeroDivisionError:\n        return {d}"


def write_toy(path, n, seed=0):
    r = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n):
            p, c = toy_example(r)
            f.write(json.dumps({"prompt": p, "code": c}) + "\n")
    print(f"wrote {n} examples to {path}")
