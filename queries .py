"""
queries.py  --  MEMBER 3: Query algorithms, benchmark and self-test

Contains:
    * range_search   (rectangle query, skips quadrants that miss the query)
    * radius_search  (bounding-box range search + exact distance filter)
    * nearest        (best-first k-nearest search using a priority queue)
    * random_query, experiment, run_benchmark   (quadtree vs linear scan, writes CSV)
    * run_self_test  (randomised correctness check of every operation)

The search functions take the tree as their first argument.

Complexity notation used in the comments above each function:
    n = number of places stored / supplied      k = number of results returned
    N = number of quadtree nodes (about n/c for evenly spread data; measured ~0.37n at n = 100,000)
    c = leaf capacity (default 8)               D = max_depth (default 20)
    d = depth of the leaf reached (d <= D; about log4(n/c) when places are evenly spread)
    m = number of queries / repetitions
"""
from __future__ import annotations

import heapq
import math
import random
from time import perf_counter_ns
from typing import List, Optional

from models import (CATEGORIES, KM_PER_DEG, LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, Place, Rect,
                    SearchResult, distance_km, generate_places, linear_search, nearest_linear,
                    radius_linear, same_ids)
from quadtree import QuadTree, build_tree


# ==============================================================================
# Search algorithms
# ==============================================================================

# Time complexity: O(d + b + k) typical for a small window; O(N + n) = O(n) worst case.
#   d = depth of the path to the window, b = leaves only partly covered by the window,
#   k = matches. Worst case happens when the window covers the whole map.
# Justification: each popped node costs O(1) plus 4 rectangle-overlap tests for its
# children. A child whose region does not overlap the window is never pushed, so its whole
# subtree (and every place in it) is skipped without being touched. Leaves that are reached
# cost O(c) point tests each. The work is therefore proportional to the part of the tree
# the window overlaps, which grows with the window area -- the benchmark shows ~35x speed-up
# over linear scan at 0.1% map area, shrinking to below 1x at 50% area.
def range_search(tree: QuadTree, q: Rect, category: Optional[str] = None) -> SearchResult:
    """Rectangle query; whole quadrants that miss q are skipped. category=None means all."""
    r = SearchResult()
    t0 = perf_counter_ns()
    x0, y0, x1, y1 = q.min_x, q.min_y, q.max_x, q.max_y
    out = r.matches
    visited = comps = 0
    stack = [tree.root]
    while stack:
        node = stack.pop()
        visited += 1
        kids = node.children
        if kids is None:
            for p in node.places:
                comps += 1
                if x0 <= p.lng <= x1 and y0 <= p.lat <= y1 and (category is None or p.category == category):
                    out.append(p)
        else:
            for c in kids:
                comps += 1
                b = c.bounds
                if b.min_x <= x1 and b.max_x >= x0 and b.min_y <= y1 and b.max_y >= y0:
                    stack.append(c)
    r.nodes_visited, r.comparisons = visited, comps
    r.nanos = perf_counter_ns() - t0
    return r


# Time complexity: same as range_search, O(d + b + k') typical and O(n) worst case, where
#   k' = places inside the bounding box (about 4/pi = 1.27x the places inside the circle).
# Justification: one range_search over the circle's bounding box (inflated by 1%), then a
# filter pass that calls the O(1) distance_km once per candidate: O(k') extra work.
# Computing dlat / dlng is O(1).
def radius_search(tree: QuadTree, lat: float, lng: float, radius_km: float,
                  category: Optional[str] = None) -> SearchResult:
    """Circle query: bounding-box range search, then an exact distance filter."""
    dlat = radius_km / KM_PER_DEG * 1.01
    dlng = radius_km / (KM_PER_DEG * math.cos(math.radians(lat))) * 1.01
    r = range_search(tree, Rect(lng - dlng, lat - dlat, lng + dlng, lat + dlat), category)
    r.matches = [p for p in r.matches if distance_km(lat, lng, p.lat, p.lng) <= radius_km]
    return r


# Time complexity: in practice about O(log n + k log k) for fixed c; worst case O(n log n).
# Justification: it is a best-first search on a min-heap. Every heappush / heappop costs
# O(log H) where H is the heap size. A quadrant's min_dist_sq is a lower bound for every
# place inside it, so the first k Place objects popped are exactly the k nearest, and
# quadrants farther away than the k-th neighbour are never expanded -- only about the path
# to the query point plus the nearby quadrants (d + k nodes) are processed, each pushing at
# most 4 children or c places. Worst case: if fewer than k places match the category
# filter, no early exit is possible, so all n places are pushed and popped (n * O(log n)).
def nearest(tree: QuadTree, lat: float, lng: float, k: int,
            category: Optional[str] = None) -> List[Place]:
    """Best-first search on min-distance to each quadrant. Degree-space Euclidean distance."""
    out: List[Place] = []
    counter = 0
    heap = [(0.0, counter, tree.root)]
    while heap and len(out) < k:
        _, _, item = heapq.heappop(heap)
        if isinstance(item, Place):
            out.append(item)
        elif item.children is None:
            for p in item.places:
                if category is None or p.category == category:
                    counter += 1
                    heapq.heappush(heap, ((p.lat - lat) ** 2 + (p.lng - lng) ** 2, counter, p))
        else:
            for c in item.children:
                counter += 1
                heapq.heappush(heap, (c.bounds.min_dist_sq(lng, lat), counter, c))
    return out


# ==============================================================================
# Evaluation (benchmark)
# ==============================================================================

# Time complexity: O(1)
# Justification: two random numbers and one Rect creation; independent of n.
def random_query(rnd: random.Random, w: float, h: float) -> Rect:
    x = LNG_MIN + rnd.random() * ((LNG_MAX - LNG_MIN) - w)
    y = LAT_MIN + rnd.random() * ((LAT_MAX - LAT_MIN) - h)
    return Rect(x, y, x + w, y + h)


# Time complexity: O(n log n + m * n), m = number of queries.
# Justification: generate_places is O(n); build_tree is O(n log n); check_invariants is
# O(N + n * d). Each of the m range queries costs the tree search (small) plus
# linear_search O(n) for the comparison, so the baseline dominates: m * O(n). The 50
# nearest checks add 50 * (tree search + nearest_linear O(n log n)) = O(n log n) since 50
# is a constant. Summing: O(n log n + m * n).
def experiment(name: str, n: int, area_frac: float, queries: int, csv_rows: List[str]) -> None:
    places = generate_places(n, 42)

    t0 = perf_counter_ns()
    tree = build_tree(places)
    build_ms = (perf_counter_ns() - t0) / 1e6

    inv = tree.check_invariants()
    if inv is not None:
        raise RuntimeError(f"Quadtree invariant broken: {inv}")

    w = (LNG_MAX - LNG_MIN) * math.sqrt(area_frac)
    h = (LAT_MAX - LAT_MIN) * math.sqrt(area_frac)
    rnd = random.Random(99)
    q_nanos = l_nanos = q_cmp = l_cmp = matches = 0
    correct = True
    for i in range(queries):
        q = random_query(rnd, w, h)
        cat = "Hospital" if i % 4 == 0 else None           # mix of all-category and filtered
        a = range_search(tree, q, cat)
        b = linear_search(places, q, cat)
        q_nanos += a.nanos
        l_nanos += b.nanos
        q_cmp += a.comparisons
        l_cmp += b.comparisons
        matches += len(a.matches)
        if not same_ids(a, b):
            correct = False

    for _ in range(50):                                     # nearest-k correctness
        lat = LAT_MIN + rnd.random() * (LAT_MAX - LAT_MIN)
        lng = LNG_MIN + rnd.random() * (LNG_MAX - LNG_MIN)
        x = nearest(tree, lat, lng, 5, "Hospital")
        y = nearest_linear(places, lat, lng, 5, "Hospital")
        if len(x) != len(y):
            correct = False
            continue
        for p, q2 in zip(x, y):
            if abs(math.hypot(p.lat - lat, p.lng - lng) - math.hypot(q2.lat - lat, q2.lng - lng)) > 1e-12:
                correct = False

    q_us, l_us = q_nanos / 1000.0 / queries, l_nanos / 1000.0 / queries
    speedup = l_us / q_us if q_us else float("inf")
    first = f"{n:,}" if name == "size" else f"{area_frac * 100:.1f}"
    print(f"{first:<9} {build_ms:<9.0f} {tree.height():<7} {tree.node_count():<9} {q_us:<11.2f} "
          f"{l_us:<11.2f} {speedup:<9.1f} {q_cmp // queries:<12} {l_cmp // queries:<12} "
          f"{'YES' if correct else 'NO':<8}")
    csv_rows.append(f"{name},{n},{area_frac * 100:.1f},{build_ms:.0f},{tree.height()},{tree.node_count()},"
                    f"{q_us:.2f},{l_us:.2f},{speedup:.1f},{q_cmp // queries},{l_cmp // queries},"
                    f"{matches / queries:.1f},{'YES' if correct else 'NO'}")


# Time complexity: O(E * (n_max log n_max + m * n_max)) with E = 7 experiments, i.e.
#   O(n log n + m * n) for the largest dataset n_max = 100,000.
# Justification: it simply calls experiment() 7 times (3 dataset sizes + 4 query sizes);
# the runs on 100,000 places dominate. Writing the CSV is O(E), one line per experiment.
def run_benchmark(csv_path: str = "results_quadtree.csv", queries: int = 300) -> None:
    # Time complexity: O(1)
    # Justification: formats a fixed-length header string with 10 column labels.
    def header(first: str) -> str:
        return (f"{first:<9} {'build ms':<9} {'height':<7} {'nodes':<9} {'QuadTree us':<11} "
                f"{'Linear us':<11} {'speedup':<9} {'QuadTree cmp':<12} {'Linear cmp':<12} {'correct':<8}")

    rows = ["experiment,n,query_area_pct,build_ms,height,nodes,quadtree_us,linear_us,speedup,"
            "quadtree_comparisons,linear_comparisons,avg_matches,correct"]

    print("=== Experiment 1: effect of dataset size (query = 1% of map area) ===")
    print(header("n"))
    for n in (1_000, 10_000, 100_000):
        experiment("size", n, 0.01, queries, rows)

    print()
    print("=== Experiment 2: effect of query size (n = 100,000) ===")
    print(header("area %"))
    for frac in (0.001, 0.01, 0.10, 0.50):
        experiment("query", 100_000, frac, queries, rows)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nSaved: {csv_path}")


# ==============================================================================
# Self-test
# ==============================================================================

# Time complexity: O(q * n log n), q = 100 nearest checks (the dominant part), n = 5,200.
# Justification: build_tree is O(n log n); 200 range checks each cost O(n) because of
# linear_search; 100 nearest checks each cost O(n log n) because of nearest_linear's sort;
# 50 radius checks cost O(n) each; 5,200 deletes cost O(c * d) each; check_invariants is
# run twice at O(N + n * d). All sizes are fixed, so the actual running time is constant
# (a few seconds), but this is how it scales if n or the number of checks were increased.
def run_self_test() -> bool:
    ok = True

    # Time complexity: O(1)
    # Justification: one boolean AND and one print call, independent of the data size.
    def check(label: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    rnd = random.Random(5)
    places = generate_places(5000, 42)
    # 200 places sharing one coordinate: would recurse forever without max_depth
    places += [Place(5000 + i, f"Dup #{i}", "ATM", 13.0, 80.2) for i in range(200)]
    tree = build_tree(places, capacity=4)
    check("build with 200 coincident places (no infinite subdivision)", tree.size() == len(places))
    check("invariants after build", tree.check_invariants() is None)

    check("insert outside map bounds is rejected", tree.insert(Place(-1, "x", "ATM", 50.0, 10.0)) is False)

    good = True
    w, h = (LNG_MAX - LNG_MIN) * 0.1, (LAT_MAX - LAT_MIN) * 0.1
    for i in range(200):
        q = random_query(rnd, w, h)
        cat = CATEGORIES[i % len(CATEGORIES)] if i % 2 else None
        good &= same_ids(range_search(tree, q, cat), linear_search(places, q, cat))
    check("200 range queries match linear scan (with/without category)", good)

    good = True
    for _ in range(100):
        lat = LAT_MIN + rnd.random() * (LAT_MAX - LAT_MIN)
        lng = LNG_MIN + rnd.random() * (LNG_MAX - LNG_MIN)
        cat = rnd.choice([None] + CATEGORIES)
        a, b = nearest(tree, lat, lng, 7, cat), nearest_linear(places, lat, lng, 7, cat)
        good &= len(a) == len(b) and all(
            abs(math.hypot(p.lat - lat, p.lng - lng) - math.hypot(q.lat - lat, q.lng - lng)) < 1e-12
            for p, q in zip(a, b))
    check("100 nearest-7 queries match linear scan (with/without category)", good)

    good = True
    for _ in range(50):
        lat = LAT_MIN + 0.05 + rnd.random() * 0.4
        lng = LNG_MIN + 0.03 + rnd.random() * 0.29
        cat = rnd.choice([None] + CATEGORIES)
        got = sorted(p.id for p in radius_search(tree, lat, lng, 3.0, cat).matches)
        want = sorted(p.id for p in radius_linear(places, lat, lng, 3.0, cat))
        good &= got == want
    check("50 radius queries (3 km) match linear scan", good)

    nodes_before = tree.node_count()
    rnd.shuffle(places)
    removed, kept = places[:3500], places[3500:]
    check("delete 3500 places", all(tree.delete(p) for p in removed) and tree.size() == len(kept))
    check("delete of an unknown id returns False", tree.delete(removed[0]) is False)
    check("invariants after deletes", tree.check_invariants() is None)
    check("tree shrinks after deletes", tree.node_count() < nodes_before)
    q = Rect(LNG_MIN, LAT_MIN, LNG_MAX, LAT_MAX)
    check("queries still correct after deletes", same_ids(range_search(tree, q), linear_search(kept, q)))

    for p in kept:
        tree.delete(p)
    check("deleting everything collapses to a single node", tree.node_count() == 1 and tree.height() == 1)
    return ok
