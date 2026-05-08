"""Problem definitions, templates, param domains, and dispatchers.

Every problem lives in PROBLEMS: one dict per problem with backend,
params, param_domains, optional is_valid_params (rejection-sampling
predicate), template, and model (pycsp only). Dispatchers route to
csp_solver or sms_solver based on the ``backend`` field.
"""

from typing import Any

from . import csp_solver, sms_solver
from .config import ProblemResult
from .log import get_logger

log = get_logger(__name__)


def _graph_tmpl(constraint: str) -> str:
    """Build a pysms template: constraint sentence + standard graph return line."""
    return constraint + (
        "\n\nReturn the graph as a list of edges (u, v) with "
        '0 <= u < v < {vertices}, or state "UNSATISFIABLE" if no graph exists.'
    )


def _is_valid_graph_params(p: dict) -> bool:
    """Reject param combinations that would be mathematically ill-defined.

    Replaces the old _repair_params silent-mutation approach with explicit
    rejection: degrees must fit [0, V-1], edge counts must be ordered, etc.
    """
    if "vertices" not in p:
        return True
    v = p["vertices"]
    max_edges = v * (v - 1) // 2
    for k in ("min_degree", "max_degree", "delta_low", "Delta_upp"):
        if k in p and not (0 <= p[k] <= v - 1):
            return False
    if "min_degree" in p and "max_degree" in p and p["min_degree"] > p["max_degree"]:
        return False
    if "delta_low" in p and "Delta_upp" in p and p["delta_low"] > p["Delta_upp"]:
        return False
    for lo_key, hi_key in (("min_edges", "max_edges"), ("num_edges_low", "num_edges_upp")):
        if lo_key in p and not (0 <= p[lo_key] <= max_edges):
            return False
        if hi_key in p and not (0 <= p[hi_key] <= max_edges):
            return False
        if lo_key in p and hi_key in p and p[lo_key] > p[hi_key]:
            return False
    for k in ("max_clique", "max_independent_set", "max_chromatic_number",
             "min_chromatic_number", "min_connectivity", "min_girth",
             "k", "clique_size"):
        if k in p and not (1 <= p[k] <= v):
            return False
    if "min_chromatic_number" in p and "max_chromatic_number" in p:
        if p["min_chromatic_number"] > p["max_chromatic_number"]:
            return False
    if "num_cliques" in p and "clique_size" in p:
        if p["num_cliques"] * p["clique_size"] > v:
            return False
    return True


def _is_valid_ramsey(p: dict) -> bool:
    """Ramsey: r, s in [3, n]."""
    return 3 <= p["r"] <= p["n"] and 3 <= p["s"] <= p["n"]


def low_autocorrelation_bound(n: int) -> int:
    """Decision-variant threshold: we ask whether a length-n binary sequence
    exists with sum-of-squared-autocorrelations at most this bound. Tightened
    from n^1.3 to ~0.10*n^2 to push the threshold near the Bernasconi
    conjecture region (~0.18*n^2 is empirically optimal energy), so SAT
    instances require near-optimal search rather than greedy descent and
    UNSAT instances require infeasibility proofs."""
    return max(4, round(0.10 * n * n))


PROBLEMS: dict[str, dict[str, Any]] = {
    # -- PyCSP3 problems (each has a unique return format) -----------------------
    "queens": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (2, 27)},
        "template": (
            "Place {n} queens on a {n}\u00d7{n} chessboard such that no two queens "
            "attack each other. Queens attack along rows, columns, and diagonals."
            "\n\nReturn a list of {n} integers where the i-th integer is the column "
            'position (0 to {n_minus_1}) of the queen in row i, or state "UNSATISFIABLE" '
            "if no solution exists."
        ),
    },
    "golomb": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (5, 10)},
        "template": (
            "Find a Golomb ruler with {n} marks. A Golomb ruler is a set of {n} "
            "integers (marks) such that all pairwise distances between marks are "
            "unique. The first mark should be at position 0."
            "\n\nReturn a list of {n} integers in increasing order representing the "
            'mark positions, or state "UNSATISFIABLE" if no solution exists.'
        ),
    },
    "magic_sequence": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (2, 16)},
        # Only n in {2,3,6} are UNSAT (3 hits in domain), so cap at unsat=3.
        "template": (
            "Find a magic sequence of length {n}. A magic sequence is a sequence "
            "x[0], x[1], ..., x[{n_minus_1}] where each x[i] equals the count of "
            "how many times i appears in the sequence."
            '\n\nReturn a list of {n} integers, or state "UNSATISFIABLE" if no '
            "solution exists."
        ),
    },
    "all_interval": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (2, 14)},
        # All n>=2 are SAT in the all-interval problem (no UNSAT in our domain).
        "template": (
            "Find an all-interval series of size {n}. This is a permutation of 0 "
            "to {n_minus_1} such that the absolute differences between consecutive "
            "elements are also a permutation of 1 to {n_minus_1}."
            "\n\nReturn a list of {n} integers forming a valid permutation, or state "
            '"UNSATISFIABLE" if no solution exists.'
        ),
    },
    "pigeons": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (3, 24)},
        "template": (
            "Place {n} pigeons into {n_minus_1} holes such that each hole contains "
            "at most one pigeon. Each pigeon must be assigned to exactly one hole "
            "(numbered 0 to {n_minus_2})."
            "\n\nReturn a list of {n} integers where the i-th integer is the hole "
            'assigned to pigeon i, or state "UNSATISFIABLE" if no solution exists.'
        ),
    },
    "ramsey": {
        "backend": "pycsp",
        "params": {"n": int, "r": int, "s": int},
        "param_domains": {"n": (5, 14), "r": (3, 4), "s": (3, 4)},
        "is_valid_params": _is_valid_ramsey,
        "template": (
            "Find a 2-coloring of the edges of the complete graph K_{n} such that "
            "there is no monochromatic red clique of size {r} and no monochromatic "
            "blue clique of size {s}. Each edge is colored either 0 (red) or 1 (blue)."
            "\n\nReturn a list of {num_edges} integers (0 or 1) representing the "
            "colors of edges listed in lexicographic order of (i,j) for i<j, or "
            'state "UNSATISFIABLE" if no such coloring exists.'
        ),
    },
    "graceful_graph": {
        "backend": "pycsp",
        "params": {"k": int, "p": int},
        "param_domains": {"k": (2, 3), "p": (3, 6)},
        "template": (
            "Find a graceful labeling for the graph G_{{{k},{p}}}: {p} disjoint K_{k} "
            "cliques (numbered 0 through {p_minus_1}), where each pair of consecutive "
            "cliques (g, g+1) is connected by {k} edges that link vertex i of clique g "
            "to vertex i of clique g+1 for each i in 0..{k_minus_1}. The graph has "
            "{n_nodes} vertices and {n_edges} edges in total."
            "\n\nA graceful labeling assigns each vertex a unique label from 0 to "
            "{n_edges} such that all edge labels |label(u) - label(v)| are distinct."
            "\n\nReturn a list of {n_nodes} integers giving the vertex labels, "
            "ordered by clique then vertex within clique (so the i-th block of {k} "
            "consecutive entries gives the labels for clique i), or state "
            '"UNSATISFIABLE" if no such labeling exists.'
        ),
    },
    "sudoku": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (2, 4)},
        "partial_assignment_fraction": 0.10,
        "template": (
            "Fill a Sudoku grid with block size {n} (so the full grid is "
            "{n_squared}x{n_squared}, containing {n_squared} rows, {n_squared} "
            "columns, and {n_squared} non-overlapping {n}x{n} blocks). Each row, "
            "column, and block must contain every integer from 1 to {n_squared} "
            "exactly once."
            "\n\nReturn a list of {total_cells} integers (the grid in row-major "
            "order: cell at row i, column j is at index i*{n_squared}+j), or state "
            '"UNSATISFIABLE" if no solution exists.'
        ),
    },
    "knight_tour": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (5, 8)},
        "template": (
            "Find an open knight's tour on an {n}x{n} chessboard that starts at "
            "cell 0 (row 0, column 0). A knight's tour visits every cell exactly "
            "once via knight moves; cells are numbered in row-major order so cell "
            "k is at row k//{n}, column k%{n}. A knight move between cells a and b "
            "changes the row by 1 and column by 2, or row by 2 and column by 1."
            "\n\nReturn a list of {total_cells} integers giving the cell visited at "
            "each step (a permutation of 0..{total_cells_minus_1} starting with 0), "
            'or state "UNSATISFIABLE" if no such tour exists.'
        ),
    },
    "langford": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (6, 18)},
        "template": (
            "Construct a Langford sequence L(2,{n}): a sequence of length "
            "{kn_total} containing exactly 2 copies of each integer from 1 to "
            "{n}, such that the two occurrences of each value v are exactly v+1 "
            "positions apart (so copies of 1 are 1 position apart, copies of 2 "
            "are 2 positions apart, etc.). Example: L(2,3) has solution "
            "[2, 3, 1, 2, 1, 3]."
            "\n\nReturn seq as a list of {kn_total} integers in 1..{n}, or "
            'state "UNSATISFIABLE" if no such sequence exists.'
        ),
    },
    "costas_array": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (5, 13)},
        "template": (
            "Find a Costas array of order {n}: place {n} marks on an {n}x{n} "
            "grid, one per row and one per column, such that all "
            "{n}*({n_minus_1})/2 displacement vectors (dr, dc) between pairs of "
            "marks (with dc > 0) are pairwise distinct."
            "\n\nReturn x as a list of {n} integers in 0..{n_minus_1}, where "
            "x[c] is the row of the mark in column c (permutation), or state "
            '"UNSATISFIABLE" if no such array exists.'
        ),
    },
    "low_autocorrelation": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (8, 24)},
        "template": (
            "Find a binary sequence of length {n} over the alphabet {{-1, +1}} "
            "with low aperiodic autocorrelation: specifically, the sum over "
            "k=1..{n_minus_1} of C_k^2, where C_k = sum_{{i=0}}^{{n-k-1}} "
            "seq[i]*seq[i+k], must be at most {autocorr_bound}."
            "\n\nReturn seq as a list of {n} integers, each -1 or +1, or state "
            '"UNSATISFIABLE" if no such sequence exists.'
        ),
    },
    "latin_square_completion": {
        "backend": "pycsp",
        "params": {"n": int, "density": int, "seed": int},
        "param_domains": {"n": (7, 10), "density": (30, 50), "seed": (1000, 99999)},
        "template": (
            "Complete the partial Latin square of order {n}. The grid is "
            "{n}x{n}; some cells are pre-filled with values from {{0, ..., {n_minus_1}}}, "
            "others are empty (denoted -1). Fill every empty cell so that each "
            "of the {n} values appears exactly once in every row and exactly "
            "once in every column."
            "\n\nThe pre-filled clues (flat row-major; cell (i,j) at index i*{n}+j; -1 = empty):\n{clues_flat}"
            "\n\nReturn the completed grid as a flat list of {n_squared} integers in "
            "row-major order (cell (i,j) at index i*{n}+j), or state \"UNSATISFIABLE\"."
        ),
    },
    "magic_square": {
        "backend": "pycsp",
        "params": {"n": int, "density": int, "seed": int},
        "param_domains": {"n": (4, 6), "density": (0, 30), "seed": (1000, 99999)},
        "template": (
            "Construct an n x n magic square of order {n}: arrange the integers "
            "1..{n_squared} so each value appears exactly once and the sum of "
            "every row, every column, and both main diagonals equals {magic_constant}."
            "\n\nPre-filled clues (flat row-major; cell (i,j) at index i*{n}+j; 0 = empty):\n{clues_flat}"
            "\n\nReturn the magic square as a flat list of {n_squared} integers in "
            "row-major order (cell (i,j) at index i*{n}+j), or state \"UNSATISFIABLE\"."
        ),
    },
    "bimagic_square": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (4, 9), "seed": (1000, 99999)},
        "template": (
            "Construct a bimagic square of order {n}: an n x n grid filled with "
            "the integers 1..{n_squared}, each used exactly once, such that every "
            "row, every column, and both main diagonals sum to the magic constant "
            "{magic_constant}, AND the squares of the entries (each x[i][j]^2) "
            "ALSO satisfy this property — every row of squares, every column of "
            "squares, and both diagonals of squares sum to the bimagic constant "
            "{bimagic_constant}."
            "\n\nReturn the grid as a flat list of {n_squared} integers in "
            "row-major order (cell (i,j) at index i*{n}+j), or state \"UNSATISFIABLE\"."
        ),
    },
    "pandiagonal_magic": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (4, 9), "seed": (1000, 99999)},
        "template": (
            "Construct a pandiagonal magic square of order {n}: an n x n grid "
            "filled with 1..{n_squared}, each used exactly once, where every row, "
            "every column, and ALL n broken diagonals (in both directions, "
            "wrapping around the torus) sum to the magic constant {magic_constant}. "
            "(The two main diagonals are special cases of broken diagonals.) "
            "Such squares exist for n in {{1, 4, 5, 7, 8, 9, ...}} but NOT for "
            "n in {{2, 3, 6}}."
            "\n\nReturn the grid as a flat list of {n_squared} integers in "
            "row-major order (cell (i,j) at index i*{n}+j), or state \"UNSATISFIABLE\"."
        ),
    },
    "schur_partition": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "seed": int},
        "param_domains": {"n": (10, 50), "k": (2, 4), "seed": (1000, 99999)},
        "template": (
            "Color the integers 1, 2, ..., {n} with {k} colors (numbered 0 to "
            "{k_minus_1}) such that no monochromatic Schur triple exists: for "
            "every (a, b, c) with 1 <= a <= b <= {n} and c = a + b <= {n}, the "
            "three integers a, b, c must NOT all share the same color. (Note "
            "a = b is allowed.) Such a coloring exists for n at most the Schur "
            "number S({k}); known values are S(1)=1, S(2)=4, S(3)=13, S(4)=44, "
            "S(5)>=160."
            "\n\nReturn a flat list of {n} integers in {{0, ..., {k_minus_1}}} "
            "where the i-th entry (0-indexed) gives the color of integer i+1, "
            "or state \"UNSATISFIABLE\"."
        ),
    },
    "kirkman_triple_system": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (9, 27), "seed": (1000, 99999)},
        "is_valid_params": lambda p: p.get("n", 0) % 6 == 3,
        "template": (
            "Construct a Kirkman triple system of order {n} (n must satisfy n ≡ 3 "
            "mod 6). This is a Steiner triple system S(2, 3, {n}) — a collection "
            "of m = {block_count} triples (3-element subsets) of {{0, ..., {n_minus_1}}} "
            "such that every pair of elements appears in exactly one triple — "
            "WITH THE ADDITIONAL property that the m triples can be partitioned "
            "into {n_classes} 'parallel classes', where each parallel class is "
            "itself a partition of {{0, ..., {n_minus_1}}} into {blocks_per_class} "
            "disjoint triples (the n elements are exactly covered, no overlaps)."
            "\n\nReturn the system as a flat 0/1 vector of length m*({n} + {n_classes}) "
            "= {total_len}: the first m*{n} = {block_inc_len} entries form the "
            "block-incidence matrix (entry b*{n}+v is 1 iff vertex v is in block b), "
            "and the remaining m*{n_classes} = {class_inc_len} entries form the "
            "block-class assignment matrix (entry m*{n} + b*{n_classes} + cls is "
            "1 iff block b is in parallel class cls). State \"UNSATISFIABLE\" if "
            "no such system exists."
        ),
    },
    "vertex_cover": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "density": int, "seed": int},
        "param_domains": {"n": (25, 60), "k": (8, 25), "density": (15, 35), "seed": (1000, 99999)},
        "template": (
            "Decide whether the following random graph G with {n} vertices "
            "(numbered 0 to {n_minus_1}) has a vertex cover of size at most "
            "{k}: a subset S of vertices such that every edge of G has at "
            "least one endpoint in S, and |S| <= {k}. NP-complete."
            "\n\nEdges of G ({edge_count} edges):\n{edges_text}"
            "\n\nReturn the cover as a flat 0/1 vector of length {n} (entry i "
            "is 1 iff vertex i is in S). The number of 1s must be at most {k}. "
            "State \"UNSATISFIABLE\" if no such cover exists."
        ),
    },
    "max_clique": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "density": int, "seed": int},
        "param_domains": {"n": (30, 80), "k": (5, 12), "density": (40, 70), "seed": (1000, 99999)},
        "template": (
            "Decide whether the following random graph G with {n} vertices "
            "(numbered 0 to {n_minus_1}) contains a clique of size at least "
            "{k}: a subset of {k} vertices, all pairwise adjacent. NP-hard."
            "\n\nEdges of G ({edge_count} edges):\n{edges_text}"
            "\n\nReturn the clique as a flat 0/1 vector of length {n} (entry i "
            "is 1 iff vertex i is in the clique). The number of 1s must be at "
            "least {k}. State \"UNSATISFIABLE\" if no such clique exists."
        ),
    },
    "graph_k_coloring": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "density": int, "seed": int},
        "param_domains": {"n": (20, 60), "k": (3, 4), "density": (10, 25), "seed": (1000, 99999)},
        "template": (
            "Decide whether the following random graph G with {n} vertices "
            "(numbered 0 to {n_minus_1}) admits a proper {k}-coloring (each "
            "vertex assigned a color from {{0, 1, ..., {k_minus_1}}} such that "
            "no two adjacent vertices share a color). NP-complete decision."
            "\n\nEdges of G ({edge_count} edges):\n{edges_text}"
            "\n\nReturn the coloring as a flat list of {n} integers in "
            "{{0, ..., {k_minus_1}}}, or state \"UNSATISFIABLE\" if no proper "
            "{k}-coloring exists."
        ),
    },
    "hamilton_cycle": {
        "backend": "pycsp",
        "params": {"n": int, "density": int, "seed": int},
        "param_domains": {"n": (15, 30), "density": (15, 35), "seed": (1000, 99999)},
        "template": (
            "Decide whether the following random graph G with {n} vertices "
            "(numbered 0 to {n_minus_1}) contains a Hamiltonian cycle — a "
            "cyclic ordering visiting every vertex exactly once where every "
            "consecutive pair (and the pair (last, first)) is an edge of G. "
            "NP-complete."
            "\n\nEdges of G ({edge_count} edges):\n{edges_text}"
            "\n\nReturn the Hamiltonian cycle as a flat list of {n} distinct "
            "integers in [0, {n}) giving the cycle order (with implicit wrap-"
            "around between last and first), starting at vertex 0. State "
            "\"UNSATISFIABLE\" if no Hamiltonian cycle exists."
        ),
    },
    "max_independent_set": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "density": int, "seed": int},
        "param_domains": {"n": (20, 50), "k": (5, 15), "density": (15, 35), "seed": (1000, 99999)},
        "template": (
            "Decide whether the following random graph G with {n} vertices "
            "(numbered 0 to {n_minus_1}) has an independent set of size at "
            "least {k} (a subset of vertices, no two of which are adjacent). "
            "NP-hard."
            "\n\nEdges of G ({edge_count} edges):\n{edges_text}"
            "\n\nReturn an independent set as a flat 0/1 vector of length {n} "
            "(entry i is 1 iff vertex i is in the set). The number of 1s must "
            "be at least {k}. State \"UNSATISFIABLE\" if no such set exists."
        ),
    },
    "antimagic_square": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (4, 8), "seed": (1000, 99999)},
        "template": (
            "Construct an antimagic square of order {n}: an n×n grid filled with "
            "the integers 1..{n_squared} (each used exactly once) such that the "
            "{two_n_plus_2} line sums (n rows, n cols, both main diagonals) form "
            "{two_n_plus_2} CONSECUTIVE integers (all distinct, with max - min = "
            "{two_n_plus_1})."
            "\n\nReturn the grid + sums as a flat list of {total_len} integers: "
            "first {n_squared} are the grid (row-major; cell (i,j) at index i*{n}+j), "
            "next {two_n_plus_2} are the sums (rows 0..{n_minus_1}, then cols "
            "0..{n_minus_1}, then main diagonal, then anti-diagonal). State "
            "\"UNSATISFIABLE\" if no such square exists."
        ),
    },
    "social_golfers": {
        "backend": "pycsp",
        "params": {"n_groups": int, "group_size": int, "n_weeks": int, "seed": int},
        "param_domains": {"n_groups": (3, 8), "group_size": (3, 5), "n_weeks": (3, 9), "seed": (1000, 99999)},
        "template": (
            "Social Golfers (CSPLib problem 10): schedule {n_players} = {n_groups}×"
            "{group_size} golfers over {n_weeks} weeks. Each week, the {n_players} "
            "golfers are partitioned into {n_groups} groups of {group_size} players "
            "each. Constraint: NO PAIR of golfers may be in the same group on more "
            "than one week. SAT iff a valid schedule exists; classic NP-hard "
            "scheduling problem."
            "\n\nReturn a flat list of {total_assignments} integers: entry "
            "w*{n_players} + p gives the group (in 0..{n_groups_minus_1}) of "
            "golfer p in week w. State \"UNSATISFIABLE\" if no schedule exists."
        ),
    },
    "debruijn": {
        "backend": "pycsp",
        "params": {"b": int, "n": int, "seed": int},
        "param_domains": {"b": (2, 4), "n": (3, 7), "seed": (1000, 99999)},
        "template": (
            "Construct a De Bruijn sequence B({b}, {n}): a cyclic sequence of "
            "length {seq_len} = {b}^{n} over the alphabet {{0, ..., {b_minus_1}}} "
            "such that every n-length subsequence (read cyclically) appears "
            "EXACTLY ONCE. Existence is guaranteed (de Bruijn 1946) but finding "
            "one at scale is non-trivial."
            "\n\nReturn the sequence as a flat list of {seq_len} integers in "
            "{{0, ..., {b_minus_1}}}, or state \"UNSATISFIABLE\"."
        ),
    },
    "number_partitioning": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "seed": int},
        "param_domains": {"n": (10, 30), "k": (3, 6), "seed": (1000, 99999)},
        "is_valid_params": lambda p: (p.get("n", 0) * (p.get("n", 0) + 1) // 2) % p.get("k", 1) == 0,
        "template": (
            "Partition the integers {{1, 2, ..., {n}}} into {k} disjoint subsets, "
            "each summing to the same value {target}. (The total sum is {n}*"
            "({n}+1)/2 = {total_sum}, so the target is {total_sum}/{k} = {target}.) "
            "NP-hard partition problem."
            "\n\nReturn a flat list of {n} integers in {{0, ..., {k_minus_1}}} "
            "where the i-th entry (0-indexed) gives the subset assignment of "
            "integer i+1. State \"UNSATISFIABLE\" if no such partition exists."
        ),
    },
    "non_transitive_dice": {
        "backend": "pycsp",
        "params": {"n": int, "m": int, "seed": int},
        "param_domains": {"n": (3, 5), "m": (4, 7), "seed": (1000, 99999)},
        "template": (
            "Design {n} cyclically intransitive dice, each with {m} faces showing "
            "values in {{0, ..., {d_minus_1}}}. The dice must satisfy: die i "
            "'beats' die (i+1) mod {n} in head-to-head face comparison — i.e., "
            "for every i in [0, {n}): #{{(a,b) : face(i,a) > face((i+1) mod {n},b)}}"
            " > {half_pairs}, out of {total_pairs} possible (a,b) face pairs. "
            "Famous problem (Efron's dice)."
            "\n\nReturn a flat list of {total_faces} integers: entry i*{m}+j is "
            "the j-th face value (0-indexed) of die i (faces sorted ascending). "
            "State \"UNSATISFIABLE\" if no such dice exist."
        ),
    },
    "hadamard": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (5, 47), "seed": (1000, 99999)},
        "is_valid_params": lambda p: p.get("n", 0) % 2 == 1,
        "template": (
            "Find two binary sequences x and y of length {n} (with n odd), each "
            "over the alphabet {{-1, +1}}, such that:\n"
            "  (a) Sum of x equals 1, Sum of y equals 1\n"
            "  (b) For every k in 1..{half_n}, the cyclic cross-correlation\n"
            "      Sum_{{i=0..{n_minus_1}}} x[i] * x[(i+k) mod {n}]  +  "
            "Sum_{{i=0..{n_minus_1}}} y[i] * y[(i+k) mod {n}]   equals -2.\n"
            "Such a pair is called a 'Legendre pair' of length n; existence is "
            "tied to number-theoretic properties of n."
            "\n\nReturn the concatenation x ++ y as a flat list of {two_n} "
            "integers (first {n} are x, next {n} are y), or state \"UNSATISFIABLE\"."
        ),
    },
    "bibd": {
        "backend": "pycsp",
        "params": {"v": int, "k": int, "ld": int, "seed": int},
        "param_domains": {"v": (6, 16), "k": (3, 5), "ld": (1, 4), "seed": (1000, 99999)},
        "is_valid_params": lambda p: (
            p.get("k", 0) > 1 and p.get("v", 0) > p.get("k", 0)
            and (p.get("ld", 0) * p.get("v", 0) * (p.get("v", 0) - 1)) % (p.get("k", 0) * (p.get("k", 0) - 1)) == 0
            and (p.get("ld", 0) * (p.get("v", 0) - 1)) % (p.get("k", 0) - 1) == 0
        ),
        "template": (
            "Construct a Balanced Incomplete Block Design BIBD({v}, {k}, {ld}): "
            "an v×b binary incidence matrix x (where v={v} points and "
            "b={b} blocks) such that:\n"
            "  - Every row sums to r={r} (each point in r blocks)\n"
            "  - Every column sums to k={k} (each block has k points)\n"
            "  - For every pair of distinct rows i₁, i₂: the inner product "
            "Sum_j x[i₁][j] * x[i₂][j] = λ={ld} (every pair of points appears "
            "in exactly λ common blocks)\n"
            "Existence requires Fisher's inequality (b ≥ v) and integrality of "
            "b, r."
            "\n\nReturn the {v}×{b} matrix as a flat list of {vb} integers in "
            "row-major order (cell (i,j) at index i*{b}+j), or state \"UNSATISFIABLE\"."
        ),
    },
    "ortholatin": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (3, 12), "seed": (1000, 99999)},
        "template": (
            "Construct a pair of orthogonal Latin squares of order {n} "
            "(also called a Greco-Latin square): two n×n grids X and Y, each "
            "filled with values from {{0, 1, ..., {n_minus_1}}}, such that:\n"
            "  - X is a Latin square (every row and every column is a permutation "
            "of 0..{n_minus_1})\n"
            "  - Y is a Latin square\n"
            "  - X and Y are orthogonal: the {nsq} pairs (X[i][j], Y[i][j]) for "
            "i,j in [0, {n}) are all distinct\n"
            "Famously, no such pair exists for n=2 or n=6 (Tarry's theorem); "
            "they exist for every other n ≥ 3."
            "\n\nReturn the concatenation X.flatten() ++ Y.flatten() as a flat "
            "list of {two_nsq} integers (first {nsq} are X row-major, next "
            "{nsq} are Y row-major), or state \"UNSATISFIABLE\"."
        ),
    },
    "quasigroup_idempotent": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (5, 12), "seed": (1000, 99999)},
        "template": (
            "Construct a quasigroup of order {n} satisfying QG3 idempotence "
            "and the involution property. Specifically: an n×n grid x with "
            "values in {{0, ..., {n_minus_1}}} such that:\n"
            "  - Every row and every column is a permutation of 0..{n_minus_1} "
            "(Latin square)\n"
            "  - x[i][i] = i for all i in [0, {n}) (idempotent diagonal)\n"
            "  - x[ x[i][j] ][ x[j][i] ] = i for all i, j in [0, {n}) (QG3)\n"
            "The third constraint involves nested indexing (the indices "
            "themselves are values from x). For n in [5, 12], few solutions "
            "exist and search is non-trivial."
            "\n\nReturn the grid as a flat list of {nsq} integers in row-major "
            "order (cell (i,j) at index i*{n}+j), or state \"UNSATISFIABLE\"."
        ),
    },
    "sidon_set": {
        "backend": "pycsp",
        "params": {"n": int, "m": int, "seed": int},
        "param_domains": {"n": (5, 16), "m": (20, 100), "seed": (1000, 99999)},
        "template": (
            "Find a Sidon (B_2) set of size {n} contained in {{1, 2, ..., {m}}}: "
            "pick {n} distinct integers x_1 < x_2 < ... < x_{n} with each "
            "x_i in [1, {m}], such that all C({n}, 2) = {pair_count} pairwise "
            "sums x_i + x_j (i < j) are pairwise distinct. Equivalently, all "
            "non-zero pairwise differences are distinct. The largest Sidon set "
            "in [1, m] has size approximately sqrt(m); existence at given (n, m) "
            "is a hard combinatorial problem."
            "\n\nReturn the set as a flat list of {n} strictly-increasing "
            "integers, or state \"UNSATISFIABLE\" if no such set exists."
        ),
    },
    "van_der_waerden": {
        "backend": "pycsp",
        "params": {"n": int, "k": int, "L": int, "seed": int},
        "param_domains": {"n": (8, 60), "k": (2, 3), "L": (3, 5), "seed": (1000, 99999)},
        "template": (
            "Color the integers 1, 2, ..., {n} with {k} colors (numbered 0 to "
            "{k_minus_1}) such that NO monochromatic arithmetic progression of "
            "length {L} exists: there is no a >= 1, d >= 1 with "
            "a + ({L_minus_1})*d <= {n} where the {L} positions a, a+d, ..., "
            "a+({L_minus_1})*d all share the same color. Such a coloring exists "
            "iff {n} < W({k}, {L}). Known values: W(2,3)=9, W(2,4)=35, W(3,3)=27, "
            "W(2,5)=178."
            "\n\nReturn a flat list of {n} integers in {{0, ..., {k_minus_1}}}, "
            "where the i-th entry (0-indexed) gives the color of integer i+1, "
            "or state \"UNSATISFIABLE\"."
        ),
    },
    "magic_hexagon": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (2, 5), "seed": (1000, 99999)},
        "template": (
            "A magic hexagon of order n is a hexagonal grid with side n "
            "containing 3n^2 - 3n + 1 = {hex_cells} cells, each filled with a "
            "distinct integer from 1 to {hex_cells}, such that the sums along "
            "all 3*(2n-1) = {hex_rows} rows in the three directions of a "
            "hexagonal grid (top-bottom, NW-SE, NE-SW) are equal. Cells are "
            "numbered row-by-row from the top, where row k (k in [0, 2n-1)) "
            "has length n + min(k, 2n-2-k), giving the standard hexagonal "
            "shape: lengths n, n+1, ..., 2n-1, ..., n+1, n."
            "\n\nReturn a flat list of {hex_cells} integers (the cell values "
            "in row-major order), or state \"UNSATISFIABLE\". Note: famously, "
            "magic hexagons exist only for n=1 (trivial) and n=3 (unique up "
            "to symmetry); n=2 and n>=4 are UNSAT."
        ),
    },
    "mutilated_chessboard": {
        "backend": "pycsp",
        "params": {"n": int, "seed": int},
        "param_domains": {"n": (6, 14), "seed": (1000, 99999)},
        "template": (
            "Cover the n x n board (n = {n}) with two opposite same-color corners "
            "removed (cells (0,0) and ({n_minus_1},{n_minus_1})) using 1x2 "
            "dominoes. Each domino covers exactly two horizontally or vertically "
            "adjacent cells; every non-removed cell must be covered by exactly "
            "one domino, and the removed cells must not be covered."
            "\n\nReturn a flat list of {dom_total} 0/1 integers indicating which "
            "domino positions are used. The first {dom_h} entries are horizontal "
            "dominoes indexed by (i, j) at i*({n_minus_1})+j for i in [0,{n}), "
            "j in [0,{n_minus_1}) (covering cells (i,j) and (i,j+1)). The last "
            "{dom_v} entries are vertical dominoes indexed by (i, j) at "
            "{dom_h}+i*{n}+j for i in [0,{n_minus_1}), j in [0,{n}) (covering "
            "cells (i,j) and (i+1,j))."
            "\n\nReturn the placement vector or state \"UNSATISFIABLE\"."
        ),
    },
    "steiner_triple_system": {
        "backend": "pycsp",
        "params": {"n": int},
        "param_domains": {"n": (9, 21)},
        "template": (
            "A Steiner triple system S(2,3,{n}) on the set V = {{0, 1, ..., {n_minus_1}}} "
            "is a collection of 3-element subsets (triples) such that every pair "
            "of distinct elements of V appears in exactly one triple. With n={n}, "
            "the system has m = {block_count} triples (when one exists)."
            "\n\nReturn the system as a flat 0/1 incidence list of length m*n = "
            "{incidence_len}, where index i*{n}+v is 1 iff vertex v is in triple i. "
            "Equivalently, every {n}-element block of the list is the indicator "
            "vector of one triple (exactly 3 ones). State \"UNSATISFIABLE\" if no "
            "such system exists."
        ),
    },
    # -- PySMS problems (all share the standard graph return line) ----------------
    "pysms_graph_builder": {
        "backend": "pysms",
        "params": {
            "vertices": int, "delta_low": int, "num_edges_low": int,
            "num_edges_upp": int, "Delta_upp": int,
            "max_chromatic_number": int, "min_chromatic_number": int,
        },
        "param_domains": {
            "vertices": (8, 14), "delta_low": (1, 3),
            "num_edges_low": (6, 18), "num_edges_upp": (10, 22),
            "Delta_upp": (2, 5), "max_chromatic_number": (2, 4),
            "min_chromatic_number": (2, 3),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl("{graph_builder_prompt}"),
    },
    "pysms_combined_graph": {
        "backend": "pysms",
        "params": {
            "vertices": int, "min_degree": int, "max_degree": int,
            "min_edges": int, "max_edges": int, "max_clique": int,
            "max_independent_set": int, "max_chromatic_number": int,
            "min_chromatic_number": int, "min_connectivity": int,
            "min_girth": int, "k": int, "maximal_triangle_free": bool,
            "num_cliques": int, "clique_size": int,
        },
        "param_domains": {
            "vertices": (10, 16), "min_degree": (1, 3), "max_degree": (3, 5),
            "min_edges": (10, 24), "max_edges": (14, 30), "max_clique": (2, 4),
            "max_independent_set": (2, 5), "max_chromatic_number": (2, 4),
            "min_chromatic_number": (2, 3), "min_connectivity": (1, 2),
            "min_girth": (3, 5), "k": (3, 5), "maximal_triangle_free": [False],
            "num_cliques": (1, 2), "clique_size": (3, 4),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl("{combined_prompt}"),
    },
    "pysms_min_degree": {
        "backend": "pysms",
        "params": {"vertices": int, "min_degree": int},
        "param_domains": {"vertices": (8, 18), "min_degree": (1, 6)},
        "is_valid_params": _is_valid_graph_params,
        # UNSAT corner (min_degree > n-1) is unreachable because _repair_params
        # clamps degree at vertices-1. Target SAT only.
        "template": _graph_tmpl("Generate a graph with {vertices} vertices where the minimum degree is at least {min_degree}."),
    },
    "pysms_degree_bounds": {
        "backend": "pysms",
        "params": {"vertices": int, "min_degree": int, "max_degree": int},
        "param_domains": {"vertices": (8, 16), "min_degree": (1, 4), "max_degree": (3, 6)},
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl("Generate a graph with {vertices} vertices where every vertex has degree between {min_degree} and {max_degree}."),
    },
    "pysms_num_edges_bounds": {
        "backend": "pysms",
        "params": {"vertices": int, "min_edges": int, "max_edges": int},
        "param_domains": {"vertices": (8, 18), "min_edges": (8, 36), "max_edges": (12, 44)},
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl("Generate a graph with {vertices} vertices where the number of edges is between {min_edges} and {max_edges}."),
    },
    "pysms_min_connectivity": {
        "backend": "pysms",
        "params": {"vertices": int, "min_connectivity": int},
        "param_domains": {"vertices": (8, 18), "min_connectivity": (1, 5)},
        "is_valid_params": _is_valid_graph_params,
        # UNSAT corner (min_connectivity > n-1) unreachable because
        # _repair_params clamps min_connectivity at vertices. Target SAT only.
        "template": _graph_tmpl("Generate a graph with {vertices} vertices with vertex-connectivity at least {min_connectivity}."),
    },
    "pysms_min_girth": {
        "backend": "pysms",
        "params": {"vertices": int, "min_girth": int},
        "param_domains": {"vertices": (8, 18), "min_girth": (3, 8)},
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl("Generate a graph with {vertices} vertices where the girth (shortest cycle length) is at least {min_girth}."),
    },
    "pysms_mtf": {
        "backend": "pysms",
        "params": {"vertices": int},
        "param_domains": {"vertices": (6, 20)},
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl("Generate a maximal triangle-free graph with {vertices} vertices."),
    },
    "pysms_contains_cliques": {
        "backend": "pysms",
        "params": {"vertices": int, "num_cliques": int, "clique_size": int},
        "param_domains": {"vertices": (10, 16), "num_cliques": (1, 4), "clique_size": (3, 5)},
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a graph with {vertices} vertices that consists of exactly "
            "{num_cliques} vertex-disjoint clique(s), each of size {clique_size}. "
            "Every vertex must belong to one of these cliques, and the only edges "
            "in the graph are those within these cliques."
        ),
    },
    # -- New compound types (all require edges — no trivial solutions) ------------
    "pysms_clique_coloring": {
        "backend": "pysms",
        "params": {"vertices": int, "max_clique": int, "max_chromatic_number": int, "min_degree": int},
        "param_domains": {
            "vertices": (8, 16), "max_clique": (2, 5),
            "max_chromatic_number": (2, 5), "min_degree": (1, 4),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a graph with {vertices} vertices where the maximum clique "
            "size is at most {max_clique}, the chromatic number is at most "
            "{max_chromatic_number}, and every vertex has degree at least {min_degree}."
        ),
    },
    "pysms_independent_connectivity": {
        "backend": "pysms",
        "params": {"vertices": int, "max_independent_set": int, "min_connectivity": int},
        "param_domains": {
            "vertices": (8, 16), "max_independent_set": (2, 6),
            "min_connectivity": (1, 4),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a graph with {vertices} vertices where the maximum "
            "independent set size is at most {max_independent_set} and the "
            "vertex-connectivity is at least {min_connectivity}."
        ),
    },
    "pysms_girth_degree": {
        "backend": "pysms",
        "params": {"vertices": int, "min_girth": int, "min_degree": int, "max_degree": int},
        "param_domains": {
            "vertices": (8, 16), "min_girth": (4, 8),
            "min_degree": (2, 4), "max_degree": (3, 6),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a graph with {vertices} vertices where the girth "
            "(shortest cycle) is at least {min_girth} and every vertex has "
            "degree between {min_degree} and {max_degree}."
        ),
    },
    "pysms_chromatic_girth": {
        "backend": "pysms",
        "params": {"vertices": int, "max_chromatic_number": int, "min_girth": int, "min_edges": int},
        "param_domains": {
            "vertices": (8, 16), "max_chromatic_number": (2, 5),
            "min_girth": (4, 7), "min_edges": (8, 22),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a graph with {vertices} vertices that is colorable with "
            "at most {max_chromatic_number} colors, has girth (shortest cycle) "
            "at least {min_girth}, and has at least {min_edges} edges."
        ),
    },
    "pysms_ramsey": {
        "backend": "pysms",
        "params": {"vertices": int, "clique_avoid": int, "indset_avoid": int},
        "param_domains": {
            "vertices": (12, 25), "clique_avoid": (3, 5), "indset_avoid": (3, 5),
        },
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a graph with {vertices} vertices that contains no clique "
            "of size {clique_avoid} and no independent set of size {indset_avoid}. "
            "(A Ramsey-witness graph: such graphs exist iff vertices < R(clique_avoid, indset_avoid).)"
        ),
    },
    "pysms_diameter2critical": {
        "backend": "pysms",
        "params": {"vertices": int},
        "param_domains": {"vertices": (6, 16)},
        "is_valid_params": _is_valid_graph_params,
        "template": _graph_tmpl(
            "Generate a diameter-2-critical graph on {vertices} vertices: "
            "the graph must have diameter exactly 2 (any two non-adjacent vertices "
            "share a neighbor), and removing any single edge must increase the diameter."
        ),
    },
}

def get_prompt(problem_type: str, params: dict) -> str:
    """Generate natural language prompt from template and params."""
    spec = PROBLEMS.get(problem_type)
    template = spec["template"] if spec else None
    if not template:
        if problem_type.startswith("pysms:"):
            template = (
                "Solve the pySMS problem '{problem_type}' with parameters {params}. "
                "Return a valid solution if satisfiable, or state \"UNSATISFIABLE\"."
            )
        else:
            raise ValueError(f"No template for problem type: {problem_type}")

    format_params = dict(params)
    if problem_type == "pysms_graph_builder":
        vertices = params.get("vertices", "unspecified")
        constraint_parts: list[str] = []
        if params.get("delta_low") is not None:
            constraint_parts.append(f"minimum degree >= {params['delta_low']}")
        if params.get("Delta_upp") is not None:
            constraint_parts.append(f"maximum degree <= {params['Delta_upp']}")
        if params.get("num_edges_low") is not None and params.get("num_edges_upp") is not None:
            constraint_parts.append(f"edges between {params['num_edges_low']} and {params['num_edges_upp']}")
        elif params.get("num_edges_low") is not None:
            constraint_parts.append(f"edges >= {params['num_edges_low']}")
        elif params.get("num_edges_upp") is not None:
            constraint_parts.append(f"edges <= {params['num_edges_upp']}")
        if params.get("max_chromatic_number") is not None:
            constraint_parts.append(f"chromatic number <= {params['max_chromatic_number']}")
        if params.get("min_chromatic_number") is not None:
            constraint_parts.append(f"chromatic number >= {params['min_chromatic_number']}")
        if constraint_parts:
            format_params["graph_builder_prompt"] = (
                f"Generate a graph with {vertices} vertices that satisfies: "
                + ", ".join(constraint_parts) + "."
            )
        else:
            format_params["graph_builder_prompt"] = f"Generate a graph with {vertices} vertices."
    if problem_type == "pysms_combined_graph":
        vertices = params.get("vertices", "unspecified")
        constraint_parts: list[str] = []
        if params.get("min_degree") is not None:
            constraint_parts.append(f"minimum degree >= {params['min_degree']}")
        if params.get("max_degree") is not None:
            constraint_parts.append(f"maximum degree <= {params['max_degree']}")
        if params.get("min_edges") is not None and params.get("max_edges") is not None:
            constraint_parts.append(f"edges between {params['min_edges']} and {params['max_edges']}")
        elif params.get("min_edges") is not None:
            constraint_parts.append(f"edges >= {params['min_edges']}")
        elif params.get("max_edges") is not None:
            constraint_parts.append(f"edges <= {params['max_edges']}")
        if params.get("max_clique") is not None:
            constraint_parts.append(f"maximum clique size <= {params['max_clique']}")
        if params.get("max_independent_set") is not None:
            constraint_parts.append(
                f"maximum independent set size <= {params['max_independent_set']}"
            )
        if params.get("max_chromatic_number") is not None:
            constraint_parts.append(
                f"chromatic number <= {params['max_chromatic_number']}"
            )
        if params.get("min_chromatic_number") is not None:
            constraint_parts.append(
                f"chromatic number >= {params['min_chromatic_number']}"
            )
        if params.get("min_connectivity") is not None:
            constraint_parts.append(
                f"vertex-connectivity >= {params['min_connectivity']}"
            )
        if params.get("min_girth") is not None:
            constraint_parts.append(f"girth >= {params['min_girth']}")
        if params.get("k") is not None:
            constraint_parts.append(f"C_{params['k']}-free")
        if params.get("maximal_triangle_free"):
            constraint_parts.append("maximal triangle-free")
        if params.get("num_cliques") is not None and params.get("clique_size") is not None:
            constraint_parts.append(
                f"consists of exactly {params['num_cliques']} vertex-disjoint cliques "
                f"of size {params['clique_size']} (every vertex belongs to one of "
                f"these cliques, and the only edges are within these cliques)"
            )
        if constraint_parts:
            combined_prompt = (
                f"Generate a graph with {vertices} vertices that satisfies: "
                + ", ".join(constraint_parts)
                + "."
            )
        else:
            combined_prompt = f"Generate a graph with {vertices} vertices."
        format_params["combined_prompt"] = combined_prompt
    if "n" in params:
        n_val = params["n"]
        format_params["n_minus_1"] = n_val - 1
        format_params["n_minus_2"] = n_val - 2
        format_params["num_edges"] = n_val * (n_val - 1) // 2
        if problem_type == "sudoku":
            format_params["n_squared"] = n_val * n_val
            format_params["total_cells"] = (n_val * n_val) * (n_val * n_val)
        if problem_type == "knight_tour":
            format_params["total_cells"] = n_val * n_val
            format_params["total_cells_minus_1"] = n_val * n_val - 1
        if problem_type == "langford":
            format_params["kn_total"] = 2 * n_val
        if problem_type == "low_autocorrelation":
            format_params["autocorr_bound"] = low_autocorrelation_bound(n_val)
    if "k" in params and "p" in params:
        k, p = params["k"], params["p"]
        format_params["n_nodes"] = k * p
        format_params["n_edges"] = ((k * (k - 1)) * p) // 2 + k * (p - 1)
        format_params["p_minus_1"] = p - 1
    if "k" in params:
        format_params["k_minus_1"] = params["k"] - 1
    if "chromatic" in params:
        format_params["chromatic_minus_one"] = params["chromatic"] - 1

    if problem_type == "latin_square_completion":
        from .csp_solver import _latin_clues
        n = params["n"]
        format_params["n_squared"] = n * n
        clues_flat = _latin_clues(n, params["density"], params["seed"])
        format_params["clues_flat"] = str(clues_flat)
    if problem_type == "magic_square":
        from .csp_solver import _magic_clues
        n = params["n"]
        format_params["n_squared"] = n * n
        format_params["magic_constant"] = n * (n * n + 1) // 2
        clues_flat = _magic_clues(n, params["density"], params["seed"])
        format_params["clues_flat"] = str(clues_flat)
    if problem_type == "steiner_triple_system":
        from math import comb
        n = params["n"]
        m = comb(n, 2) // 3
        format_params["block_count"] = m
        format_params["incidence_len"] = m * n
    if problem_type in ("bimagic_square", "pandiagonal_magic"):
        n = params["n"]
        format_params["n_squared"] = n * n
        format_params["magic_constant"] = n * (n * n + 1) // 2
        if problem_type == "bimagic_square":
            format_params["bimagic_constant"] = n * (n * n + 1) * (2 * n * n + 1) // 6
    if problem_type == "schur_partition":
        format_params["k_minus_1"] = params["k"] - 1
    if problem_type == "kirkman_triple_system":
        n = params["n"]
        m = n * (n - 1) // 6
        n_classes = (n - 1) // 2
        blocks_per_class = n // 3
        format_params["block_count"] = m
        format_params["n_classes"] = n_classes
        format_params["blocks_per_class"] = blocks_per_class
        format_params["block_inc_len"] = m * n
        format_params["class_inc_len"] = m * n_classes
        format_params["total_len"] = m * (n + n_classes)
    if problem_type in ("graph_k_coloring", "hamilton_cycle", "max_independent_set",
                         "vertex_cover", "max_clique"):
        from .csp_solver import _random_graph_edges
        n = params["n"]
        edges = _random_graph_edges(n, params["density"], params["seed"])
        format_params["edge_count"] = len(edges)
        format_params["edges_text"] = ", ".join(f"({u},{v})" for u, v in edges)
        if problem_type == "graph_k_coloring":
            format_params["k_minus_1"] = params["k"] - 1
    if problem_type == "antimagic_square":
        n = params["n"]
        format_params["n_squared"] = n * n
        format_params["two_n_plus_2"] = 2 * n + 2
        format_params["two_n_plus_1"] = 2 * n + 1
        format_params["total_len"] = n * n + 2 * n + 2
    if problem_type == "social_golfers":
        ng, gs, nw = params["n_groups"], params["group_size"], params["n_weeks"]
        format_params["n_players"] = ng * gs
        format_params["n_groups_minus_1"] = ng - 1
        format_params["total_assignments"] = ng * gs * nw
    if problem_type == "debruijn":
        b, n = params["b"], params["n"]
        format_params["seq_len"] = b ** n
        format_params["b_minus_1"] = b - 1
    if problem_type == "number_partitioning":
        n, k = params["n"], params["k"]
        total = n * (n + 1) // 2
        format_params["total_sum"] = total
        format_params["target"] = total // k
        format_params["k_minus_1"] = k - 1
    if problem_type == "non_transitive_dice":
        n, m = params["n"], params["m"]
        format_params["d_minus_1"] = 2 * m - 1
        format_params["half_pairs"] = m * m // 2
        format_params["total_pairs"] = m * m
        format_params["total_faces"] = n * m
    if problem_type == "hadamard":
        n = params["n"]
        format_params["half_n"] = (n - 1) // 2
        format_params["two_n"] = 2 * n
    if problem_type == "bibd":
        v, k, ld = params["v"], params["k"], params["ld"]
        b = (ld * v * (v - 1)) // (k * (k - 1))
        r = (ld * (v - 1)) // (k - 1)
        format_params["b"] = b
        format_params["r"] = r
        format_params["vb"] = v * b
    if problem_type == "ortholatin":
        n = params["n"]
        format_params["nsq"] = n * n
        format_params["two_nsq"] = 2 * n * n
    if problem_type == "quasigroup_idempotent":
        n = params["n"]
        format_params["nsq"] = n * n
    if problem_type == "sidon_set":
        from math import comb
        format_params["pair_count"] = comb(params["n"], 2)
    if problem_type == "van_der_waerden":
        format_params["k_minus_1"] = params["k"] - 1
        format_params["L_minus_1"] = params["L"] - 1
    if problem_type == "magic_hexagon":
        n = params["n"]
        format_params["hex_cells"] = 3 * n * n - 3 * n + 1
        format_params["hex_rows"] = 3 * (2 * n - 1)
    if problem_type == "mutilated_chessboard":
        n = params["n"]
        format_params["dom_h"] = n * (n - 1)
        format_params["dom_v"] = n * (n - 1)
        format_params["dom_total"] = 2 * n * (n - 1)

    return template.format(problem_type=problem_type, params=params, **format_params)


def get_param_domain(problem_name: str, param_key: str) -> tuple[int, int] | list[Any]:
    """Resolve the configured sampling domain for a problem parameter."""
    if problem_name not in PROBLEMS:
        raise KeyError(f"Unknown problem: {problem_name}")
    spec = PROBLEMS[problem_name]
    spec_domains = spec.get("param_domains", {})
    if param_key in spec_domains:
        return spec_domains[param_key]
    log.warning(
        "No domain for param %r of problem %r; falling back to (1, 12)",
        param_key,
        problem_name,
    )
    return (1, 12)




def list_problems() -> list[str]:
    return list(PROBLEMS.keys())


def get_spec(name: str) -> dict:
    if name not in PROBLEMS:
        raise KeyError(f"Unknown problem: {name}")
    return PROBLEMS[name]


def get_backend(name: str) -> str:
    spec = PROBLEMS.get(name)
    if spec is None:
        raise KeyError(f"Unknown problem: {name}")
    return spec["backend"]


def solve(name: str, **params) -> ProblemResult:
    spec = PROBLEMS.get(name)
    if spec is None:
        raise KeyError(f"Unknown problem: {name}")
    if spec["backend"] == "pycsp":
        return csp_solver.solve(name, **params)
    return sms_solver.solve(name, **params)


def verify(name: str, params: dict, solution: Any) -> tuple[bool, str]:
    spec = PROBLEMS.get(name)
    if spec is None:
        raise KeyError(f"Unknown problem: {name}")
    if spec["backend"] == "pycsp":
        return csp_solver.verify(name, params, solution)
    return sms_solver.verify(name, params, solution)


def get_search_space(name: str, **params) -> int:
    if name not in PROBLEMS:
        raise KeyError(f"Unknown problem: {name}")
    spec = PROBLEMS[name]
    if spec["backend"] != "pycsp":
        return -1
    return csp_solver.get_search_space(name, **params)
