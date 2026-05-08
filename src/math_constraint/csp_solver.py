"""PyCSP3 solver interface."""

import ast
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .log import get_logger, solver_timeout_seconds
from .config import ProblemResult, SolverArtifacts

log = get_logger(__name__)


def queens_model(n: int = 8) -> str:
    return f"""
from pycsp3 import *

n = {n}
q = VarArray(size=n, dom=range(n))

satisfy(
    AllDifferent(q),
    AllDifferent(q[i] + i for i in range(n)),
    AllDifferent(q[i] - i for i in range(n))
)
"""


def golomb_model(n: int = 4) -> str:
    ub = n * n
    return f"""
from pycsp3 import *

n = {n}
ub = {ub}

x = VarArray(size=n, dom=range(ub + 1))

satisfy(
    x[0] == 0,
    Increasing(x, strict=True),
    AllDifferent(x[j] - x[i] for i in range(n) for j in range(i + 1, n))
)

minimize(x[-1])
"""


def all_interval_model(n: int = 8) -> str:
    return f"""
from pycsp3 import *

n = {n}

x = VarArray(size=n, dom=range(n))
d = VarArray(size=n-1, dom=range(1, n))

satisfy(
    AllDifferent(x),
    AllDifferent(d),
    [d[i] == abs(x[i+1] - x[i]) for i in range(n-1)]
)
"""


def magic_sequence_model(n: int = 8) -> str:
    return f"""
from pycsp3 import *

n = {n}

x = VarArray(size=n, dom=range(n))

satisfy(
    [Count(x, value=i) == x[i] for i in range(n)],
    Sum(x) == n,
    Sum(i * x[i] for i in range(n)) == n
)
"""


def graceful_graph_model(k: int = 2, p: int = 4) -> str:
    n_nodes = k * p
    n_edges = ((k * (k - 1)) * p) // 2 + k * (p - 1)
    return f"""
from pycsp3 import *

k = {k}
p = {p}
n_nodes = {n_nodes}
n_edges = {n_edges}

# Build edge list for K_k x P_p graph
edges = []
# Edges within each K_k clique
for g in range(p):
    for i in range(k):
        for j in range(i + 1, k):
            edges.append((g * k + i, g * k + j))
# Edges between consecutive cliques (path structure)
for g in range(p - 1):
    for i in range(k):
        edges.append((g * k + i, (g + 1) * k + i))

# Node labels in [0, n_edges]
x = VarArray(size=n_nodes, dom=range(n_edges + 1))

# Edge labels (absolute difference of endpoint labels)
d = VarArray(size=n_edges, dom=range(1, n_edges + 1))

satisfy(
    AllDifferent(x),
    AllDifferent(d),
    [d[i] == abs(x[edges[i][0]] - x[edges[i][1]]) for i in range(n_edges)]
)
"""


def ramsey_model(n: int = 8, r: int = 3, s: int = 3) -> str:
    return f"""
from pycsp3 import *
from itertools import combinations

n = {n}
r = {r}
s = {s}

# Edge colors: 0 = red, 1 = blue
# Use upper triangle of adjacency matrix
edges = [(i, j) for i in range(n) for j in range(i+1, n)]
num_edges = len(edges)

c = VarArray(size=num_edges, dom=range(2))

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

# No red clique of size r
for clique in combinations(range(n), r):
    clique_edges = [c[edge_idx(i, j)] for i, j in combinations(clique, 2)]
    satisfy(Sum(clique_edges) >= 1)  # At least one blue edge

# No blue clique of size s
for clique in combinations(range(n), s):
    clique_edges = [c[edge_idx(i, j)] for i, j in combinations(clique, 2)]
    satisfy(Sum(clique_edges) <= len(clique_edges) - 1)  # At least one red edge
"""


def pigeons_model(n: int = 8) -> str:
    return f"""
from pycsp3 import *

n = {n}
holes = n - 1

p = VarArray(size=n, dom=range(holes))

satisfy(
    AllDifferent(p)
)
"""


def sudoku_model(n: int = 3) -> str:
    return f"""
from pycsp3 import *

n = {n}
N = n * n

x = VarArray(size=N * N, dom=range(1, N + 1))

satisfy(
    [AllDifferent(x[i * N:(i + 1) * N]) for i in range(N)],
    [AllDifferent([x[i * N + j] for i in range(N)]) for j in range(N)],
    [
        AllDifferent(
            [x[(bi * n + di) * N + (bj * n + dj)] for di in range(n) for dj in range(n)]
        )
        for bi in range(n) for bj in range(n)
    ],
)
"""


def knight_tour_model(n: int = 6) -> str:
    return f"""
from pycsp3 import *

n = {n}

x = VarArray(size=n * n, dom=range(n * n))

satisfy(
    AllDifferent(x),
    x[0] == 0,
)

pairs = [(i, i + 1) for i in range(n * n - 1)]

satisfy(
    (d1 == 1) & (d2 == 2) | (d1 == 2) & (d2 == 1)
    for i, j in pairs
    if (d1 := abs(x[i] // n - x[j] // n), d2 := abs(x[i] % n - x[j] % n))
)
"""


def langford_model(n: int = 10) -> str:
    # L(2, n): length-2n sequence where each value v in [1, n] appears twice,
    # with the two copies exactly v+1 positions apart. Expressed with a single
    # VarArray so the XCSP3 solution parser stays happy.
    return f"""
from pycsp3 import *
from functools import reduce
import operator

n = {n}
length = 2 * n

seq = VarArray(size=length, dom=range(1, n + 1))

satisfy(
    [Count(seq, value=v) == 2 for v in range(1, n + 1)],
)

# If seq[p] == v, then v's paired copy sits at p+v+1 OR p-v-1.
for v in range(1, n + 1):
    for p in range(length):
        options = [seq[p] != v]
        if p + v + 1 < length:
            options.append(seq[p + v + 1] == v)
        if p - v - 1 >= 0:
            options.append(seq[p - v - 1] == v)
        if len(options) == 1:
            satisfy(options[0])
        else:
            satisfy(reduce(operator.or_, options))
"""


def costas_array_model(n: int = 8) -> str:
    return f"""
from pycsp3 import *

n = {n}

# x[c] is the row of the mark in column c (a permutation of 0..n-1).
x = VarArray(size=n, dom=range(n))

satisfy(
    AllDifferent(x),
    [AllDifferent(x[i] - x[i + d] for i in range(n - d)) for d in range(1, n - 1)]
)
"""


def low_autocorrelation_model(n: int = 12) -> str:
    from .problems import low_autocorrelation_bound
    bound = low_autocorrelation_bound(n)
    return f"""
from pycsp3 import *

n = {n}
bound = {bound}

# seq[i] is the i-th value of the +/-1 sequence.
seq = VarArray(size=n, dom={{-1, 1}})

# c[k] is the k-th aperiodic autocorrelation (sum_{{i=0}}^{{n-k-2}} seq[i]*seq[i+k+1]).
c = VarArray(size=n - 1, dom=range(-n + 1, n))

satisfy(
    [c[k] == Sum(seq[i] * seq[i + k + 1] for i in range(n - k - 1)) for k in range(n - 1)],
    Sum(c[k] * c[k] for k in range(n - 1)) <= bound,
)
"""


_SEARCH_SPACE = {
    "queens":         lambda n, **_: n ** n,
    "golomb":         lambda n, **_: (n * n + 1) ** n,
    "all_interval":   lambda n, **_: math.factorial(n),
    "magic_sequence":  lambda n, **_: n ** n,
    "graceful_graph":  lambda k, p, **_: (((k * (k - 1)) * p) // 2 + k * (p - 1) + 1) ** (k * p),
    "ramsey":         lambda n, **_: 2 ** (n * (n - 1) // 2),
    "pigeons":        lambda n, **_: (n - 1) ** n,
    "sudoku":         lambda n, **_: (n * n) ** (n ** 4),
    "knight_tour":    lambda n, **_: math.factorial(n * n),
    "langford":       lambda n, **_: math.factorial(2 * n),
    "costas_array":   lambda n, **_: math.factorial(n),
    "low_autocorrelation": lambda n, **_: 2 ** n,
    "latin_square_completion": lambda n, **_: math.factorial(n) ** n,
    "magic_square": lambda n, **_: math.factorial(n * n),
    "steiner_triple_system": lambda n, **_: math.comb(n, 3) ** ((n * (n - 1)) // 6),
    "bimagic_square":      lambda n, **_: math.factorial(n * n),
    "pandiagonal_magic":   lambda n, **_: math.factorial(n * n),
    "schur_partition":     lambda n, k, **_: k ** n,
    "mutilated_chessboard": lambda n, **_: 2 ** (2 * n * (n - 1)),
    "van_der_waerden":     lambda n, k, **_: k ** n,
    "magic_hexagon":       lambda n, **_: math.factorial(3 * n * n - 3 * n + 1),
    "sidon_set":           lambda n, m, **_: math.comb(m, n),
    "hadamard":            lambda n, **_: 2 ** (2 * n),
    "bibd":                lambda v, k, ld, **_: 2 ** (v * ((ld * v * (v - 1)) // (k * (k - 1)))),
    "ortholatin":          lambda n, **_: (math.factorial(n) ** (2 * n)),
    "quasigroup_idempotent": lambda n, **_: math.factorial(n) ** n,
    "antimagic_square":    lambda n, **_: math.factorial(n * n),
    "social_golfers":      lambda n_groups, group_size, n_weeks, **_: n_groups ** (n_groups * group_size * n_weeks),
    "debruijn":            lambda b, n, **_: b ** (b ** n),
    "number_partitioning": lambda n, k, **_: k ** n,
    "non_transitive_dice": lambda n, m, **_: (2 * m) ** (n * m),
    "graph_k_coloring":    lambda n, k, **_: k ** n,
    "hamilton_cycle":      lambda n, **_: math.factorial(n),
    "max_independent_set": lambda n, **_: 2 ** n,
    "vertex_cover":        lambda n, **_: 2 ** n,
    "max_clique":          lambda n, **_: 2 ** n,
    "kirkman_triple_system": lambda n, **_: math.comb(n, 3) ** ((n * (n - 1)) // 6),
}


def _latin_clues(n: int, density: int, seed: int) -> list[int]:
    """Deterministic clue grid (flat row-major; -1 = empty cell)."""
    rng = random.Random(seed)
    rows = list(range(n))
    rng.shuffle(rows)
    base = [[(rows[i] + j) % n for j in range(n)] for i in range(n)]
    cols = list(range(n))
    rng.shuffle(cols)
    base_p = [[base[i][cols[j]] for j in range(n)] for i in range(n)]
    n_clues = round(n * n * density / 100)
    cells = [(i, j) for i in range(n) for j in range(n)]
    clue_cells = set(rng.sample(cells, n_clues))
    flat = [base_p[i][j] if (i, j) in clue_cells else -1
            for i in range(n) for j in range(n)]
    return flat


def latin_square_model(n: int = 7, density: int = 40, seed: int = 0) -> str:
    """Latin square completion: n×n grid filled with {0..n-1}, each value
    appearing exactly once per row and column. density% of cells are preset
    from a randomly chosen Latin square so completion always exists.
    NP-hard search even when SAT. Flat 1D VarArray (row-major) so the
    universal verify path works without modification."""
    clues = _latin_clues(n, density, seed)
    return f"""
from pycsp3 import *

n = {n}
clues = {clues}

x = VarArray(size=n * n, dom=range(n))

satisfy(
    [AllDifferent(x[i * n:(i + 1) * n]) for i in range(n)],
    [AllDifferent([x[i * n + j] for i in range(n)]) for j in range(n)],
    [x[i * n + j] == clues[i * n + j]
        for i in range(n) for j in range(n) if clues[i * n + j] != -1],
)
"""


def _magic_clues(n: int, density: int, seed: int) -> list[int]:
    """Deterministic clue grid (flat row-major; 0 = empty cell)."""
    rng = random.Random(seed)
    bases = {
        3: [[2, 7, 6], [9, 5, 1], [4, 3, 8]],
        4: [[1, 15, 14, 4], [12, 6, 7, 9], [8, 10, 11, 5], [13, 3, 2, 16]],
        5: [[17, 24, 1, 8, 15], [23, 5, 7, 14, 16], [4, 6, 13, 20, 22],
            [10, 12, 19, 21, 3], [11, 18, 25, 2, 9]],
    }
    if density > 0 and n in bases:
        base = bases[n]
        n_clues = round(n * n * density / 100)
        cells = [(i, j) for i in range(n) for j in range(n)]
        clue_cells = set(rng.sample(cells, n_clues))
        return [base[i][j] if (i, j) in clue_cells else 0
                for i in range(n) for j in range(n)]
    return [0] * (n * n)


def magic_square_model(n: int = 4, density: int = 0, seed: int = 0) -> str:
    """Classical n×n magic square: 1..n² each exactly once, every row, column
    and both main diagonals sum to n(n²+1)/2. Optional clues from seed.
    Flat 1D VarArray for verifier consistency."""
    clues = _magic_clues(n, density, seed)
    magic = n * (n * n + 1) // 2
    return f"""
from pycsp3 import *

n = {n}
magic = {magic}
clues = {clues}

x = VarArray(size=n * n, dom=range(1, n * n + 1))

satisfy(
    AllDifferent(x),
    [Sum(x[i * n + j] for j in range(n)) == magic for i in range(n)],
    [Sum(x[i * n + j] for i in range(n)) == magic for j in range(n)],
    Sum(x[i * n + i] for i in range(n)) == magic,
    Sum(x[i * n + (n - 1 - i)] for i in range(n)) == magic,
    [x[i * n + j] == clues[i * n + j]
        for i in range(n) for j in range(n) if clues[i * n + j] != 0],
)
"""


def steiner_triple_model(n: int = 11) -> str:
    """Steiner triple system S(2,3,n): partition C(n,2) pairs of an n-set into
    m = n(n-1)/6 triples; every pair appears in exactly one triple. SAT iff
    n ≡ 1 or 3 (mod 6). Encoded as a flat n×n 0/1 incidence (whether each
    unordered pair is covered) via a flat block-by-element matrix; using
    1D VarArray for verifier consistency.

    Variable layout (flat, length m*n): cell[i*n + v] = 1 iff vertex v is
    in block i, else 0. Each block has exactly k=3 ones; each pair (u,v)
    is covered by exactly one block."""
    k = 3
    m = math.comb(n, 2) // math.comb(k, 2)
    return f"""
from pycsp3 import *
from itertools import combinations

n = {n}
m = {m}
k = {k}

x = VarArray(size=m * n, dom={{0, 1}})

def block(i):
    return [x[i * n + v] for v in range(n)]

satisfy(
    # each block has exactly k=3 elements
    [Sum(block(i)) == k for i in range(m)],
    # each unordered pair (u,v) covered by exactly one block
    [Sum(x[i * n + u] * x[i * n + v] for i in range(m)) == 1
        for u, v in combinations(range(n), 2)],
)
"""


def bimagic_square_model(n: int = 4, seed: int = 0) -> str:
    """Bimagic square: an n×n grid filled with 1..n^2 so that all rows, all
    columns, both main diagonals sum to the magic constant n(n^2+1)/2 AND the
    same is true for the SQUARES of the entries (sum to n(n^2+1)(2n^2+1)/6).
    Bimagic squares of order 8 are the smallest known; n in {4,5,6,7} are
    UNSAT; n=8 is the famous Pfeffermann square. We bracket: n=8,9 SAT;
    n=4,5,6,7 UNSAT. Seed unused but kept for interface uniformity."""
    magic = n * (n * n + 1) // 2
    bimagic = n * (n * n + 1) * (2 * n * n + 1) // 6
    return f"""
from pycsp3 import *

n = {n}
magic = {magic}
bimagic = {bimagic}

x = VarArray(size=n * n, dom=range(1, n * n + 1))

satisfy(
    AllDifferent(x),
    [Sum(x[i * n + j] for j in range(n)) == magic for i in range(n)],
    [Sum(x[i * n + j] for i in range(n)) == magic for j in range(n)],
    Sum(x[i * n + i] for i in range(n)) == magic,
    Sum(x[i * n + (n - 1 - i)] for i in range(n)) == magic,
    [Sum(x[i * n + j] * x[i * n + j] for j in range(n)) == bimagic for i in range(n)],
    [Sum(x[i * n + j] * x[i * n + j] for i in range(n)) == bimagic for j in range(n)],
    Sum(x[i * n + i] * x[i * n + i] for i in range(n)) == bimagic,
    Sum(x[i * n + (n - 1 - i)] * x[i * n + (n - 1 - i)] for i in range(n)) == bimagic,
)
"""


def pandiagonal_magic_model(n: int = 4, seed: int = 0) -> str:
    """Pandiagonal magic square: n×n with rows, cols, both main diagonals AND
    ALL broken diagonals (wrapped around the torus) summing to n(n^2+1)/2.
    Existence: requires n != 2, 3, 6 mod ... actually pandiagonal magic
    squares exist for n = 1, 4, 5, 7, 8, 9, ... and don't for n = 2, 3, 6.
    n=4: SAT (rare). n=5: SAT. n=6: UNSAT. n=7: SAT."""
    magic = n * (n * n + 1) // 2
    return f"""
from pycsp3 import *

n = {n}
magic = {magic}

x = VarArray(size=n * n, dom=range(1, n * n + 1))

satisfy(
    AllDifferent(x),
    [Sum(x[i * n + j] for j in range(n)) == magic for i in range(n)],
    [Sum(x[i * n + j] for i in range(n)) == magic for j in range(n)],
    # all n broken right-diagonals (i + d mod n)
    [Sum(x[i * n + ((i + d) % n)] for i in range(n)) == magic for d in range(n)],
    # all n broken left-diagonals (i - d mod n)
    [Sum(x[i * n + ((d - i) % n)] for i in range(n)) == magic for d in range(n)],
)
"""


def schur_partition_model(n: int = 13, k: int = 3, seed: int = 0) -> str:
    """Schur partition: assign each integer 1..n a color from 0..k-1 such
    that there is no monochromatic Schur triple (a, b, c) with a + b = c
    (a, b, c need not be distinct; a <= b < c <= n). Schur numbers S(k):
    S(1)=1, S(2)=4, S(3)=13, S(4)=44, S(5)=160. So for k=3, n<=13 SAT and
    n>=14 UNSAT; for k=4, n<=44 SAT and n>=45 UNSAT. Seed unused."""
    # x[a-1]==x[b-1] and x[b-1]==x[a+b-1] implies x[a-1]==x[a+b-1], so it suffices
    # to forbid two-of-three equalities. Wrap each comparison in parens because
    # pycsp3's | overload binds tighter than !=.
    return f"""
from pycsp3 import *

n = {n}
k = {k}

x = VarArray(size=n, dom=range(k))

satisfy(
    [(x[a - 1] != x[b - 1]) | (x[b - 1] != x[a + b - 1])
        for a in range(1, n + 1)
        for b in range(a, n + 1)
        if a + b <= n],
)
"""


def mutilated_chessboard_model(n: int = 8, seed: int = 0) -> str:
    """Mutilated chessboard: cover an n×n board with two opposite same-color
    corners removed using 1×2 dominoes. Each domino covers exactly 2 adjacent
    cells; every non-removed cell is covered exactly once. UNSAT for ALL n
    by the parity argument: dominoes always cover one black + one white,
    but removing two same-color corners leaves an imbalance. Encoded as a
    flat 0/1 placement grid over horizontal and vertical domino positions."""
    # Number of horizontal positions: n*(n-1); vertical: same.
    nH = n * (n - 1)
    nV = n * (n - 1)
    return f"""
from pycsp3 import *

n = {n}
nH = {nH}  # horizontal dominoes: at position (i,j) for i in [0,n), j in [0,n-1)
nV = {nV}  # vertical dominoes: at position (i,j) for i in [0,n-1), j in [0,n)
removed = [(0, 0), (n - 1, n - 1)]  # both same color (n+(n-2) = even, both black/white together)
removed_set = set(removed)

# x[k] = 1 iff k-th domino is placed. Index: 0..nH-1 are horizontal, nH..nH+nV-1 are vertical.
x = VarArray(size=nH + nV, dom={{0, 1}})

def horiz(i, j): return x[i * (n - 1) + j]   # spans (i,j) and (i,j+1)
def vert(i, j):  return x[nH + i * n + j]    # spans (i,j) and (i+1,j)

def covers(r, c):
    parts = []
    if c > 0: parts.append(horiz(r, c - 1))
    if c < n - 1: parts.append(horiz(r, c))
    if r > 0: parts.append(vert(r - 1, c))
    if r < n - 1: parts.append(vert(r, c))
    return parts

satisfy(
    [Sum(covers(r, c)) == 1 for r in range(n) for c in range(n) if (r, c) not in removed_set],
    [Sum(covers(r, c)) == 0 for (r, c) in removed],
)
"""


def kirkman_triple_model(n: int = 9, seed: int = 0) -> str:
    """Kirkman triple system: a Steiner triple system S(2,3,n) that admits
    a partition of its blocks into 'parallel classes', where each parallel
    class is itself a partition of {0..n-1} into n/3 disjoint triples.
    Existence requires n ≡ 3 (mod 6). For n=9: 1 known KTS (the affine plane
    of order 3, unique up to iso). For n=15: 7 non-isomorphic KTSs (Kirkman's
    schoolgirl problem). Strictly harder to construct than a plain STS."""
    assert n % 6 == 3
    n_blocks = n * (n - 1) // 6
    n_classes = (n - 1) // 2  # parallel classes
    blocks_per_class = n // 3
    return f"""
from pycsp3 import *
from itertools import combinations

n = {n}
m = {n_blocks}
n_classes = {n_classes}
blocks_per_class = {blocks_per_class}

# x[i*n + v] = 1 iff vertex v is in block i. Length m*n.
# Plus: c[b*n_classes + cls] = 1 iff block b is in parallel class cls.
# Combined flat layout: first m*n is x, then m*n_classes is c.
total = m * n + m * n_classes
xc = VarArray(size=total, dom={{0, 1}})

def x(b, v): return xc[b * n + v]
def c(b, cls): return xc[m * n + b * n_classes + cls]

satisfy(
    # Steiner: each block has exactly 3 elements
    [Sum(x(b, v) for v in range(n)) == 3 for b in range(m)],
    # Each pair (u,v) covered by exactly one block
    [Sum(x(b, u) * x(b, v) for b in range(m)) == 1
        for u, v in combinations(range(n), 2)],
    # Each block in exactly one parallel class
    [Sum(c(b, cls) for cls in range(n_classes)) == 1 for b in range(m)],
    # Each parallel class has exactly n/3 blocks
    [Sum(c(b, cls) for b in range(m)) == blocks_per_class for cls in range(n_classes)],
    # Within each class, blocks must be disjoint (cover each vertex once):
    # for each class cls and each vertex v: sum over blocks b of c(b,cls) * x(b,v) == 1
    [Sum(c(b, cls) * x(b, v) for b in range(m)) == 1
        for cls in range(n_classes) for v in range(n)],
)
"""


def _random_graph_edges(n: int, density: int, seed: int) -> list:
    """Deterministic G(n, p) edge list."""
    rng = random.Random(seed)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if rng.random() < density / 100]


def graph_k_coloring_model(n: int = 30, k: int = 3, density: int = 15, seed: int = 0) -> str:
    """Graph k-coloring: given a random G(n, p) graph baked into the prompt,
    decide if there's a proper k-coloring (adjacent vertices get distinct
    colors). NP-complete. Hardness peak for k=3 near density ~4/n*100.
    n=30-50, density 13-20% with k=3 sits at the phase transition."""
    edges = _random_graph_edges(n, density, seed)
    return f"""
from pycsp3 import *

n, k = {n}, {k}
edges = {edges}

x = VarArray(size=n, dom=range(k))

satisfy(
    [x[i] != x[j] for (i, j) in edges],
    # symmetry break
    x[0] == 0,
)
"""


def hamilton_cycle_model(n: int = 18, density: int = 25, seed: int = 0) -> str:
    """Hamiltonian cycle: given a random G(n, p) graph, decide if a
    Hamiltonian cycle exists. NP-complete. Encoded as: find permutation
    pi of vertices such that consecutive (and wrap-around) vertices are
    edges of G. n=15-25 with density slightly above HC threshold (log n / n
    or so) sits at the boundary."""
    edges = _random_graph_edges(n, density, seed)
    edge_set = set()
    for i, j in edges:
        edge_set.add((i, j))
        edge_set.add((j, i))
    edge_set_list = sorted(edge_set)
    return f"""
from pycsp3 import *

n = {n}
edges = {edge_set_list}
edge_set = set(edges)

# x[i] is the i-th vertex in the Hamiltonian cycle ordering
x = VarArray(size=n, dom=range(n))

satisfy(
    AllDifferent(x),
    # consecutive vertices must be connected
    [(x[i], x[(i + 1) % n]) in edges for i in range(n)],
    # symmetry break: start at vertex 0
    x[0] == 0,
)
"""


def max_independent_set_model(n: int = 30, k: int = 8, density: int = 30, seed: int = 0) -> str:
    """Maximum Independent Set decision: given G(n, p), decide if there's
    an independent set of size >= k (no two chosen vertices share an edge).
    NP-hard. n=20-50 with density and k tuned near max α(G)."""
    edges = _random_graph_edges(n, density, seed)
    return f"""
from pycsp3 import *

n, k = {n}, {k}
edges = {edges}

# x[i] = 1 iff vertex i is in the independent set
x = VarArray(size=n, dom={{0, 1}})

satisfy(
    Sum(x) >= k,
    [x[i] + x[j] <= 1 for (i, j) in edges],
)
"""


def vertex_cover_model(n: int = 30, k: int = 12, density: int = 25, seed: int = 0) -> str:
    """Vertex cover decision: given G(n, p), is there a subset S of size <= k
    such that every edge has at least one endpoint in S? NP-complete."""
    edges = _random_graph_edges(n, density, seed)
    return f"""
from pycsp3 import *

n, k = {n}, {k}
edges = {edges}

# x[i] = 1 iff vertex i is in the cover
x = VarArray(size=n, dom={{0, 1}})

satisfy(
    Sum(x) <= k,
    [x[i] + x[j] >= 1 for (i, j) in edges],
)
"""


def max_clique_model(n: int = 30, k: int = 5, density: int = 50, seed: int = 0) -> str:
    """Max clique decision: given G(n, p), is there a clique of size >= k?
    NP-hard. Hardness peak at moderate density."""
    edges = _random_graph_edges(n, density, seed)
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    edge_list = sorted(edge_set)
    return f"""
from pycsp3 import *
from itertools import combinations

n, k = {n}, {k}
edges = set({edge_list})

# x[i] = 1 iff vertex i is in the clique
x = VarArray(size=n, dom={{0, 1}})

satisfy(
    Sum(x) >= k,
    # for every non-edge pair (i, j), can't have both in clique
    [x[i] + x[j] <= 1 for i in range(n) for j in range(i + 1, n)
        if (i, j) not in edges],
)
"""


def antimagic_square_model(n: int = 4, seed: int = 0) -> str:
    """Antimagic square: arrangement of 1..n^2 in n×n grid such that the 2n+2
    line sums (n rows + n cols + 2 diagonals) form 2n+2 CONSECUTIVE integers
    (all distinct, with Max - Min = 2n+1). Stronger than magic-different and
    different family from magic squares. Existence: SAT for n>=4 (small) but
    very rare; n=4 has only 18 solutions up to symmetry."""
    lb = (n * (n + 1)) // 2
    ub = (n * n * (n * n + 1)) // 2
    return f"""
from pycsp3 import *

n = {n}
lb, ub = {lb}, {ub}

# Flat: x[0..n^2-1] then y[0..2n+1]
total = n * n + 2 * n + 2
xy = VarArray(size=total, dom=range(0, ub + 1))

def cell(i, j): return xy[i * n + j]
def y(k): return xy[n * n + k]

# Constrain x domain
satisfy(
    [cell(i, j) >= 1 for i in range(n) for j in range(n)],
    [cell(i, j) <= n * n for i in range(n) for j in range(n)],
    AllDifferent([cell(i, j) for i in range(n) for j in range(n)]),
    [y(i) == Sum(cell(i, j) for j in range(n)) for i in range(n)],
    [y(n + j) == Sum(cell(i, j) for i in range(n)) for j in range(n)],
    y(2 * n) == Sum(cell(i, i) for i in range(n)),
    y(2 * n + 1) == Sum(cell(i, n - 1 - i) for i in range(n)),
    AllDifferent([y(k) for k in range(2 * n + 2)]),
    Maximum([y(k) for k in range(2 * n + 2)]) - Minimum([y(k) for k in range(2 * n + 2)]) == 2 * n + 1,
)
"""


def social_golfers_model(n_groups: int = 4, group_size: int = 4, n_weeks: int = 5, seed: int = 0) -> str:
    """Social Golfers (CSPLib prob10): schedule n_groups × group_size players
    over n_weeks rounds of golf, each week partitioning all players into groups,
    such that NO pair of players is in the same group on more than one week.
    Famous NP-hard scheduling. Boundary cases (n_groups, group_size, n_weeks)
    range from trivial (4,3,2) to hard (8,4,9), (5,4,7), (8,5,7)."""
    n_players = n_groups * group_size
    return f"""
from pycsp3 import *
from itertools import combinations

n_groups = {n_groups}
group_size = {group_size}
n_weeks = {n_weeks}
n_players = {n_players}

# g[w * n_players + p] = group of player p in week w (in 0..n_groups-1)
g = VarArray(size=n_weeks * n_players, dom=range(n_groups))

def assign(w, p): return g[w * n_players + p]

satisfy(
    # each group has exactly group_size players each week
    [Cardinality([assign(w, p) for p in range(n_players)],
                 occurrences={{grp: group_size for grp in range(n_groups)}})
        for w in range(n_weeks)],
    # no pair of players together more than once across weeks:
    # for each pair (p1, p2), Sum over weeks w of (assign(w, p1) == assign(w, p2)) <= 1
    [Sum((assign(w, p1) == assign(w, p2)) for w in range(n_weeks)) <= 1
        for p1, p2 in combinations(range(n_players), 2)],
    # symmetry breaking: week 0 has players 0..gs-1 in group 0, gs..2gs-1 in group 1, ...
    [assign(0, p) == p // group_size for p in range(n_players)],
)
"""


def debruijn_model(b: int = 2, n: int = 3, seed: int = 0) -> str:
    """De Bruijn sequence B(b, n): a cyclic sequence over alphabet of size b
    such that every n-tuple appears exactly once as a contiguous subsequence.
    Length is b^n. Always SAT (de Bruijn 1946). Hard at scale: B(2,8)=256,
    B(3,5)=243, B(2,10)=1024."""
    length = b ** n
    return f"""
from pycsp3 import *

b = {b}
n = {n}
m = {length}

# Sequence of length m
x = VarArray(size=m, dom=range(b))

# All cyclic n-tuples must be distinct
def tuple_id(i):
    return Sum(x[(i + k) % m] * (b ** (n - 1 - k)) for k in range(n))

satisfy(
    AllDifferent([tuple_id(i) for i in range(m)]),
)
"""


def number_partitioning_model(n: int = 16, k: int = 4, seed: int = 0) -> str:
    """Number partitioning: partition {1..n} into k disjoint subsets of equal
    sum. Sum is n(n+1)/2; SAT requires k | n(n+1)/2. NP-hard.
    Hardness: at the boundary of feasibility."""
    total = n * (n + 1) // 2
    if total % k != 0:
        # UNSAT trivially
        return f"""
from pycsp3 import *
n, k = {n}, {k}
x = VarArray(size=n, dom=range(k))
satisfy(x[0] != x[0])
"""
    target = total // k
    return f"""
from pycsp3 import *

n, k = {n}, {k}
target = {target}

# x[i] = which subset element i+1 belongs to (0..k-1)
x = VarArray(size=n, dom=range(k))

satisfy(
    [Sum((x[i] == j) * (i + 1) for i in range(n)) == target for j in range(k)],
    # symmetry breaking: 1 always in subset 0
    x[0] == 0,
)
"""


def non_transitive_dice_model(n: int = 3, m: int = 6, seed: int = 0) -> str:
    """Intransitive dice: design n dice, each with m faces showing values from
    [0, 2m), such that die_i 'beats' die_{i+1} (in expected face comparison)
    and die_{n-1} beats die_0 — making the 'A beats B' relation cyclic
    (rock-paper-scissors-like). Famous problem (Efron's dice).
    Hardness: existence non-trivial; hard at n>=3, m=6."""
    d = 2 * m
    return f"""
from pycsp3 import *

n = {n}
m = {m}
d = {d}

# Flat: x[i * m + j] = jth face of ith die
x = VarArray(size=n * m, dom=range(d))

def face(i, j): return x[i * m + j]

# y[i] = number of (face_i_a, face_(i+1)_b) pairs where face_i_a > face_(i+1)_b
# Total pairs = m * m
half = m * m // 2

satisfy(
    # increasing faces (symmetry break)
    [[face(i, j) <= face(i, j + 1) for j in range(m - 1)] for i in range(n)],
    # die i beats die (i+1) cyclically — strict majority
    [Sum((face(i, a) > face((i + 1) % n, b)) for a in range(m) for b in range(m)) > half
        for i in range(n)],
)
"""


def hadamard_model(n: int = 9, seed: int = 0) -> str:
    """Hadamard matrix / Legendre pair: find two ±1 sequences x, y of length
    n (odd) such that Sum(x)=1, Sum(y)=1, and for k=1..(n-1)/2 the cyclic
    cross-correlation Sum(x[i]*x[(i+k)%n]) + Sum(y[i]*y[(i+k)%n]) = -2.
    UNSAT for many odd n; the smallest unresolved cases include n=85, n=87.
    Known SAT for n in {3,5,7,9,11,13,15,...} via Legendre symbols at primes."""
    assert n % 2 == 1
    return f"""
from pycsp3 import *

n = {n}
m = (n - 1) // 2

# Flat 1D: x[0..n-1] then y[0..n-1]
x = VarArray(size=2 * n, dom={{-1, 1}})

def X(i): return x[i % n]
def Y(i): return x[n + (i % n)]

satisfy(
    Sum(X(i) for i in range(n)) == 1,
    Sum(Y(i) for i in range(n)) == 1,
    [Sum(X(i) * X(i + k + 1) for i in range(n))
     + Sum(Y(i) * Y(i + k + 1) for i in range(n)) == -2 for k in range(m)],
)
"""


def bibd_model(v: int = 7, k: int = 3, ld: int = 1, seed: int = 0) -> str:
    """Balanced Incomplete Block Design BIBD(v, k, λ). v points, blocks of
    size k, each pair of points appears in exactly λ common blocks. b and r
    derived: b = λv(v-1) / (k(k-1)), r = λ(v-1)/(k-1). Existence requires
    integrality of b,r and Fisher's inequality b≥v. Hardness: classic
    combinatorial design problem; solver-edge for v=10-15 with small λ."""
    b = (ld * v * (v - 1)) // (k * (k - 1)) if k > 1 else 0
    r = (ld * (v - 1)) // (k - 1) if k > 1 else 0
    return f"""
from pycsp3 import *
from itertools import combinations

v, b, r, k, ld = {v}, {b}, {r}, {k}, {ld}

# Flat row-major v×b binary incidence
x = VarArray(size=v * b, dom={{0, 1}})

def row(i): return [x[i * b + j] for j in range(b)]
def col(j): return [x[i * b + j] for i in range(v)]

satisfy(
    [Sum(row(i)) == r for i in range(v)],
    [Sum(col(j)) == k for j in range(b)],
    [Sum(x[i1 * b + j] * x[i2 * b + j] for j in range(b)) == ld
        for i1, i2 in combinations(range(v), 2)],
)
"""


def ortholatin_model(n: int = 6, seed: int = 0) -> str:
    """Pair of orthogonal Latin squares of order n (Greco-Latin square).
    Two Latin squares X, Y such that pairs (X[i][j], Y[i][j]) are all
    distinct (n² distinct pairs). FAMOUS UNSAT for n=2,6 (Tarry 1900);
    SAT for all n>=3 except n=6. Uses encoding: derived flat variable
    z[i*n+j] = X[i][j]*n + Y[i][j]; require AllDifferent(z)."""
    return f"""
from pycsp3 import *

n = {n}

# Flat: x = X.flatten() :: Y.flatten() (length 2*n*n)
x = VarArray(size=2 * n * n, dom=range(n))

def X(i, j): return x[i * n + j]
def Y(i, j): return x[n * n + i * n + j]

satisfy(
    # X is Latin
    [AllDifferent([X(i, j) for j in range(n)]) for i in range(n)],
    [AllDifferent([X(i, j) for i in range(n)]) for j in range(n)],
    # Y is Latin
    [AllDifferent([Y(i, j) for j in range(n)]) for i in range(n)],
    [AllDifferent([Y(i, j) for i in range(n)]) for j in range(n)],
    # Orthogonality: pairs (X[i][j], Y[i][j]) distinct (encoded as X*n+Y)
    AllDifferent([X(i, j) * n + Y(i, j) for i in range(n) for j in range(n)]),
)
"""


def quasigroup_idempotent_model(n: int = 8, seed: int = 0) -> str:
    """QuasiGroup QG3: Latin square of order n with the property
    x[x[i][j]][x[j][i]] = i for all i,j. Idempotent (x[i][i]=i). The
    nested-index Element constraint is the encoding-hard part: x[a][b]
    where a,b are themselves x-cell values requires the model to recognize
    Element. Few solutions exist; n in [5,12] is the interesting range."""
    return f"""
from pycsp3 import *

n = {n}

# Flat row-major Latin square
x = VarArray(size=n * n, dom=range(n))

def cell(i, j): return x[i * n + j]

satisfy(
    # Latin
    [AllDifferent([cell(i, j) for j in range(n)]) for i in range(n)],
    [AllDifferent([cell(i, j) for i in range(n)]) for j in range(n)],
    # Idempotent
    [cell(i, i) == i for i in range(n)],
    # Main property (V3): x[ x[i][j] ][ x[j][i] ] == i, encoded via element
    # access on flat array using arithmetic: index = cell(i,j)*n + cell(j,i)
    [x[cell(i, j) * n + cell(j, i)] == i for i in range(n) for j in range(n)],
)
"""


def sidon_set_model(n: int = 8, m: int = 50, seed: int = 0) -> str:
    """Sidon (B_2) set: choose n integers x_0 < x_1 < ... < x_{n-1} from
    {1, 2, ..., m} such that all C(n, 2) pairwise SUMS x_i + x_j (i < j)
    are distinct. (Equivalently, pairwise differences are distinct.) The
    largest Sidon set in [1, m] has size O(sqrt(m)). Many cases are open.
    Distinct from Golomb because Golomb fixes x_0=0 and requires all C(n,2)
    differences distinct + minimizes the largest mark."""
    return f"""
from pycsp3 import *

n = {n}
m = {m}

x = VarArray(size=n, dom=range(1, m + 1))

satisfy(
    Increasing(x, strict=True),
    AllDifferent([x[i] + x[j] for i in range(n) for j in range(i + 1, n)]),
)
"""


def van_der_waerden_model(n: int = 9, k: int = 2, L: int = 3, seed: int = 0) -> str:
    """Van der Waerden problem: color {1..n} with k colors such that no
    monochromatic arithmetic progression of length L exists. SAT iff
    n < W(k, L). Known: W(2,3)=9, W(2,4)=35, W(3,3)=27, W(2,5)=178.
    Encoded by enumerating all APs (a, a+d, ..., a+(L-1)d) inside [1..n]."""
    return f"""
from pycsp3 import *

n = {n}
k = {k}
L = {L}

x = VarArray(size=n, dom=range(k))

# All length-L APs starting at 1<=a<=n with step d>=1, fitting in [1..n]
APs = [tuple(a + i * d for i in range(L))
       for a in range(1, n + 1)
       for d in range(1, (n - a) // (L - 1) + 1)]

satisfy(
    [Or(x[ap[i] - 1] != x[ap[j] - 1] for i in range(L) for j in range(i + 1, L))
        for ap in APs],
)
"""


def magic_hexagon_model(n: int = 3, seed: int = 0) -> str:
    """Magic hexagon of order n: a hexagonal grid with side n has 3n²-3n+1
    cells, filled with 1..3n²-3n+1 (each used once). All 3(2n-1) rows in
    the three directions (top-bottom, NW-SE, NE-SW) must sum to the magic
    constant M = (3n²-3n+1)(3n²-3n+2) / (2(2n-1)). Famously, only n=1
    (trivial) and n=3 (unique) admit solutions; n=2 and n>=4 are UNSAT.

    Cells are numbered 0..N-1 row-major (rows of length n, n+1, ..., 2n-1,
    ..., n+1, n). Row k has length n + min(k, 2n-2-k) for k in [0, 2n-1)."""
    N = 3 * n * n - 3 * n + 1
    M_num = N * (N + 1)
    M_den = 2 * (2 * n - 1)
    if M_num % M_den != 0:
        # Magic constant non-integer — UNSAT trivially
        return f"""
from pycsp3 import *

# Magic constant non-integer for n={n}; problem is UNSAT.
x = VarArray(size={N}, dom={{0}})
satisfy(x[0] != 0)  # force UNSAT
"""
    M = M_num // M_den
    # Row layout: row k for k in [0, 2n-1), length L_k = n + k if k < n else n + (2n-2-k)
    rows_horiz = []
    idx = 0
    for k in range(2 * n - 1):
        Lk = n + k if k < n else n + (2 * n - 2 - k)
        rows_horiz.append(list(range(idx, idx + Lk)))
        idx += Lk
    # Build NW-SE and NE-SW diagonals from coordinates (q, r) using offset coords
    # Cell at (k, j) where k = row index in [0, 2n-1), j in [0, len(row[k]))
    # Convert to axial coords (q, r): r = k - n + 1, q = j - max(0, n-1-k)
    coords = {}
    for k in range(2 * n - 1):
        Lk = n + k if k < n else n + (2 * n - 2 - k)
        offset = max(0, n - 1 - k)
        for j in range(Lk):
            cell_id = rows_horiz[k][j]
            coords[cell_id] = (j - offset + (k if k < n else n - 1), k - n + 1)
    # NW-SE rows: cells with same q (but use p+r constancy actually; depends on axial)
    # Simpler: for each axis, group cells by axial coordinate.
    # Use cube coords: for hex, three coords summing to 0. NE-SW: same x; NW-SE: same y; H: same z.
    cube = {cid: (q, r, -q - r) for cid, (q, r) in coords.items()}
    rows_a = sorted([sorted(cid for cid, c in cube.items() if c[0] == v)
                     for v in {c[0] for c in cube.values()}], key=len)
    rows_b = sorted([sorted(cid for cid, c in cube.items() if c[1] == v)
                     for v in {c[1] for c in cube.values()}], key=len)
    rows_c = sorted([sorted(cid for cid, c in cube.items() if c[2] == v)
                     for v in {c[2] for c in cube.values()}], key=len)
    all_rows = rows_horiz + rows_a + rows_b + rows_c
    # Filter rows of length 1 (single-cell — automatically equal to itself, no real sum constraint
    # except the cell value itself must equal M which would over-constrain).
    # Standard magic hexagon: only rows of length >= n contribute. Use as-is.
    return f"""
from pycsp3 import *

N = {N}
M = {M}
rows = {all_rows}

x = VarArray(size=N, dom=range(1, N + 1))

satisfy(
    AllDifferent(x),
    [Sum(x[i] for i in row) == M for row in rows if len(row) >= 1],
)
"""


_MODELS: dict[str, dict[str, Any]] = {
    "queens":              {"model": queens_model},
    "golomb":              {"model": golomb_model},
    "all_interval":        {"model": all_interval_model},
    "magic_sequence":      {"model": magic_sequence_model},
    "graceful_graph":      {"model": graceful_graph_model},
    "ramsey":              {"model": ramsey_model},
    "pigeons":             {"model": pigeons_model},
    "sudoku":              {"model": sudoku_model},
    "knight_tour":         {"model": knight_tour_model},
    "langford":            {"model": langford_model},
    "costas_array":        {"model": costas_array_model},
    "low_autocorrelation": {"model": low_autocorrelation_model},
    "latin_square_completion": {"model": latin_square_model},
    "magic_square":        {"model": magic_square_model},
    "steiner_triple_system": {"model": steiner_triple_model},
    "bimagic_square":      {"model": bimagic_square_model},
    "pandiagonal_magic":   {"model": pandiagonal_magic_model},
    "schur_partition":     {"model": schur_partition_model},
    "mutilated_chessboard": {"model": mutilated_chessboard_model},
    "van_der_waerden":     {"model": van_der_waerden_model},
    "magic_hexagon":       {"model": magic_hexagon_model},
    "sidon_set":           {"model": sidon_set_model},
    "hadamard":            {"model": hadamard_model},
    "bibd":                {"model": bibd_model},
    "ortholatin":          {"model": ortholatin_model},
    "quasigroup_idempotent": {"model": quasigroup_idempotent_model},
    "antimagic_square":    {"model": antimagic_square_model},
    "social_golfers":      {"model": social_golfers_model},
    "debruijn":            {"model": debruijn_model},
    "number_partitioning": {"model": number_partitioning_model},
    "non_transitive_dice": {"model": non_transitive_dice_model},
    "graph_k_coloring":    {"model": graph_k_coloring_model},
    "hamilton_cycle":      {"model": hamilton_cycle_model},
    "max_independent_set": {"model": max_independent_set_model},
    "vertex_cover":        {"model": vertex_cover_model},
    "max_clique":          {"model": max_clique_model},
}


def _parse_xcsp3_stats(xcsp3: str) -> tuple[int, int]:
    num_vars = 0
    for match in re.finditer(r'size="\[(\d+)\]"', xcsp3):
        num_vars += int(match.group(1))
    constraints_match = re.search(r'<constraints>(.*?)</constraints>', xcsp3, re.DOTALL)
    if constraints_match:
        num_constraints = len(re.findall(r'<(?!/)(\w+)', constraints_match.group(1)))
    else:
        num_constraints = 0
    return num_vars, num_constraints


def _expand_xcsp3_values(values_str: str) -> list[int]:
    result = []
    for token in values_str.split():
        if 'x' in token:
            value, count = token.split('x')
            result.extend([int(value, 0)] * int(count))
        else:
            result.append(int(token, 0))
    return result


def _parse_solution(sol_str: str | None) -> dict[str, Any] | None:
    if not sol_str:
        return None
    result: dict[str, Any] = {}

    # Modern XCSP3 instantiation output:
    #   <list> seq[] c[] pos[][] </list>
    #   <values> v1 v2 v3   w1 w2   x1 x2 </values>
    # Variable name groups inside <list> are separated by whitespace; value
    # groups inside <values> are separated by 2+ whitespace characters.
    list_match = re.search(r'<list>\s*(.+?)\s*</list>', sol_str, re.DOTALL)
    values_match = re.search(r'<values>\s*(.+?)\s*</values>', sol_str, re.DOTALL)
    if list_match and values_match:
        var_names = re.findall(r'(\w+)(?:\s*\[\])+', list_match.group(1))
        values_str = values_match.group(1).strip()
        value_groups = re.split(r'\s{2,}', values_str)
        if len(var_names) >= 1 and len(value_groups) == len(var_names):
            for name, group in zip(var_names, value_groups):
                result[name] = _expand_xcsp3_values(group)
            return result
        if len(var_names) == 1:
            result[var_names[0]] = _expand_xcsp3_values(values_str)
            return result
        # Fallback: can't align groups, put all values under the first var.
        if var_names:
            result[var_names[0]] = _expand_xcsp3_values(values_str)
            return result

    for match in re.finditer(r'(\w+):\s*(\[[^\]]+\]|\d+)', sol_str):
        var_name = match.group(1)
        value = match.group(2)
        try:
            result[var_name] = ast.literal_eval(value)
        except Exception:
            result[var_name] = value
    return result if result else {"raw": sol_str}


def _extract_variable_name(model_code: str) -> str | None:
    match = re.search(r'(\w+)\s*=\s*VarArray', model_code)
    return match.group(1) if match else None


def get_search_space(problem_name: str, **params) -> int:
    fn = _SEARCH_SPACE.get(problem_name)
    return fn(**params) if fn else -1


def solve(problem_name: str, **params) -> ProblemResult:
    """Run pycsp3 model in subprocess, parse XCSP3 output."""
    spec = _MODELS[problem_name]
    model_code = spec["model"](**params)

    solve_code = model_code + '''
from pycsp3 import solve, solution, status, SAT, UNSAT, OPTIMUM
import json
import time

start = time.time()
result = solve()
solve_time_ms = (time.time() - start) * 1000

sol_data = {
    "status": str(status()),
    "solve_time_ms": solve_time_ms,
}
if status() in (SAT, OPTIMUM):
    sol = solution()
    if sol is not None:
        sol_data["solution"] = str(sol)
with open("solution.json", "w") as f:
    json.dump(sol_data, f)
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_file = tmpdir / "model.py"
        model_file.write_text(solve_code)

        log.debug(f"[pycsp3] Solving {problem_name} with params={params}")
        timeout_s = solver_timeout_seconds()
        try:
            result = subprocess.run(
                [sys.executable, str(model_file)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            log.warning(f"[pycsp3] Timeout: {problem_name} params={params}")
            raise RuntimeError(f"Solver timeout for {problem_name} with params {params}")
        if result.returncode != 0:
            log.error(f"[pycsp3] Failed: {result.stderr[:200]}")
            raise RuntimeError(f"Model failed: {result.stderr}")

        xml_files = list(tmpdir.glob("*.xml"))
        if not xml_files:
            raise RuntimeError("No XCSP3 file generated")
        xcsp3_content = xml_files[0].read_text()
        num_vars, num_constraints = _parse_xcsp3_stats(xcsp3_content)

        artifacts = SolverArtifacts(
            model=xcsp3_content,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        sol_file = tmpdir / "solution.json"
        if sol_file.exists():
            sol_data = json.loads(sol_file.read_text())
            status_str = sol_data.get("status", "")
            satisfiable = status_str in ("SAT", "TypeStatus.SAT", "OPTIMUM", "TypeStatus.OPTIMUM")
            solution = _parse_solution(sol_data.get("solution")) if satisfiable else None
            solve_time_ms = sol_data.get("solve_time_ms", -1)
        else:
            satisfiable = False
            solution = None
            solve_time_ms = -1

    log.info(f"[pycsp3] {problem_name}: sat={satisfiable} vars={num_vars} cons={num_constraints} time={solve_time_ms:.1f}ms")
    return ProblemResult(
        satisfiable=satisfiable,
        solution=solution,
        solve_time_ms=solve_time_ms,
        num_variables=num_vars,
        num_constraints=num_constraints,
        artifacts=artifacts,
    )


def verify(problem_name: str, params: dict, solution: Any) -> tuple[bool, str]:
    """Re-run model with solution fixed as equality constraints."""
    spec = _MODELS[problem_name]
    model_code = spec["model"](**params)
    if not model_code:
        return False, "No model code available for verification"

    var_name = _extract_variable_name(model_code)
    if not var_name:
        return False, "Could not determine variable name from model"

    verify_code = model_code + f'''
from pycsp3 import solve, status, SAT, OPTIMUM
import json

solution = json.loads({json.dumps(json.dumps(solution))})
for vname, values in (solution if isinstance(solution, dict) else {{"{var_name}": solution}}).items():
    var = globals().get(vname)
    if var is not None:
        for i, val in enumerate(values):
            satisfy(var[i] == val)

result = solve()
with open("verify_result.json", "w") as f:
    json.dump({{"status": str(status())}}, f)
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_file = tmpdir / "verify_model.py"
        model_file.write_text(verify_code)

        try:
            result = subprocess.run(
                [sys.executable, str(model_file)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=solver_timeout_seconds() or 60,
            )
        except subprocess.TimeoutExpired:
            return False, "Verification timeout"

        if result.returncode != 0:
            stderr = result.stderr.lower()
            if "unsat" in stderr or "inconsistent" in stderr:
                return False, "Solution violates constraints"
            return False, f"Verification failed: {result.stderr[:200]}"

        result_file = tmpdir / "verify_result.json"
        if result_file.exists():
            data = json.loads(result_file.read_text())
            status_str = data.get("status", "")
            if "UNSAT" in status_str:
                return False, "Solution violates constraints (UNSAT)"
            elif "SAT" in status_str or "OPTIMUM" in status_str:
                return True, "Solution verified by solver"
            else:
                return False, f"Solution invalid: solver returned {status_str}"
        else:
            return False, "Verification did not produce result"
