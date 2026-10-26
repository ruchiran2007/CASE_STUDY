"""
models.py  --  MEMBER 1: Data layer and linear-scan baselines

Contains:
    * Map constants (bounds, categories)
    * Place, Rect          (core records)
    * distance_km          (equirectangular distance)
    * generate_places      (reproducible clustered + uniform sample data)
    * SearchResult         (matches + measurements)
    * linear_search, nearest_linear, radius_linear, same_ids
      (brute-force baselines used for benchmarking and for checking correctness)

Complexity notation used in the comments above each function:
    n = number of places stored / supplied      k = number of results returned
    N = number of quadtree nodes (about n/c for evenly spread data; measured ~0.37n at n = 100,000)
    c = leaf capacity (default 8)               D = max_depth (default 20)
    d = depth of the leaf reached (d <= D; about log4(n/c) when places are evenly spread)
    m = number of queries / repetitions
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import List, Optional

# ------------------------------------------------------------------ constants
LAT_MIN, LAT_MAX = 12.80, 13.30
LNG_MIN, LNG_MAX = 80.00, 80.35
CATEGORIES = ["Hospital", "Restaurant", "ATM", "Pharmacy", "Fuel Station", "School"]
KM_PER_DEG = 111.32


# --------------------------------------------------------------------- records
@dataclass(frozen=True, slots=True)
class Place:
    """One place record: id, name, category, latitude (y), longitude (x)."""
    id: int
    name: str
    category: str
    lat: float
    lng: float

    # Time complexity: O(1)
    # Justification: formats a fixed number of fields (id, name, category, lat, lng) into one
    # string; the work does not depend on how many places exist.
    def __str__(self) -> str:
        return f"#{self.id} {self.name} [{self.category}] ({self.lat:.5f}, {self.lng:.5f})"


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle. x = longitude, y = latitude. All tests are boundary-inclusive."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    # Time complexity: O(1)
    # Justification: exactly four comparisons joined by 'and'; no loops, no dependence on n.
    def intersects(self, o: Rect) -> bool:
        return (self.min_x <= o.max_x and self.max_x >= o.min_x and
                self.min_y <= o.max_y and self.max_y >= o.min_y)

    # Time complexity: O(1)
    # Justification: exactly four comparisons joined by 'and'; no loops, no dependence on n.
    def contains(self, o: Rect) -> bool:
        return (self.min_x <= o.min_x and self.max_x >= o.max_x and
                self.min_y <= o.min_y and self.max_y >= o.max_y)

    # Time complexity: O(1)
    # Justification: at most four comparisons (a chained comparison per axis); no loops.
    def contains_point(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    # Time complexity: O(1)
    # Justification: two max() calls over three numbers, two multiplications and one addition,
    # which is a fixed number of arithmetic operations.
    def min_dist_sq(self, x: float, y: float) -> float:
        """Squared distance from (x, y) to the nearest point of this rectangle (0 if inside)."""
        dx = max(self.min_x - x, 0.0, x - self.max_x)
        dy = max(self.min_y - y, 0.0, y - self.max_y)
        return dx * dx + dy * dy


@dataclass(slots=True)
class SearchResult:
    """Matches plus the measurements used in the evaluation."""
    matches: List[Place] = field(default_factory=list)
    nodes_visited: int = 0     # quadtree nodes opened (linear scan: 0)
    comparisons: int = 0       # rectangle / point tests performed (linear scan: n)
    nanos: int = 0


# -------------------------------------------------------------------- geometry
# Time complexity: O(1)
# Justification: a fixed number of float operations (radians, cos, hypot, multiply);
# each is constant time, so the cost does not depend on the data size.
def distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Equirectangular approximation, accurate enough for a city-sized area."""
    x = math.radians(lng2 - lng1) * math.cos(math.radians((lat1 + lat2) / 2.0))
    y = math.radians(lat2 - lat1)
    return 6371.0088 * math.hypot(x, y)


# ------------------------------------------------------------------ data maker
# Time complexity: O(n) time, O(n) space
# Justification: setting up the 25 cluster centres is constant work. The main loop runs n
# times and each iteration does O(1) work (a few random draws, two clamps, one Place
# creation, one append). Nothing inside the loop depends on n, so the total is n * O(1).
def generate_places(n: int, seed: int = 42) -> List[Place]:
    """Reproducible sample places inside a Chennai-like box: 70% clustered, 30% uniform."""
    rnd = random.Random(seed)
    clusters = []
    for _ in range(25):
        clusters.append((LAT_MIN + rnd.random() * (LAT_MAX - LAT_MIN),
                         LNG_MIN + rnd.random() * (LNG_MAX - LNG_MIN),
                         0.004 + rnd.random() * 0.02))
    out: List[Place] = []
    for i in range(n):
        if rnd.random() < 0.70:
            clat, clng, spread = clusters[rnd.randrange(len(clusters))]
            lat = clat + rnd.gauss(0, 1) * spread
            lng = clng + rnd.gauss(0, 1) * spread
        else:
            lat = LAT_MIN + rnd.random() * (LAT_MAX - LAT_MIN)
            lng = LNG_MIN + rnd.random() * (LNG_MAX - LNG_MIN)
        lat = min(LAT_MAX, max(LAT_MIN, lat))
        lng = min(LNG_MAX, max(LNG_MIN, lng))
        cat = CATEGORIES[rnd.randrange(len(CATEGORIES))]
        out.append(Place(i, f"{cat} #{i}", cat, lat, lng))
    return out


# ------------------------------------------------------------------- baselines
# Time complexity: O(n) -- best, average and worst case are all Theta(n)
# Justification: the loop visits every place exactly once and does a constant number of
# comparisons on it (4 range tests + 1 category test). Appending a match is amortised O(1).
# There is no early exit, so the cost never drops below n tests; this is the baseline that
# the quadtree is compared against.
def linear_search(places: List[Place], q: Rect, category: Optional[str] = None) -> SearchResult:
    """Baseline: test every place. O(n)."""
    r = SearchResult()
    t0 = perf_counter_ns()
    x0, y0, x1, y1 = q.min_x, q.min_y, q.max_x, q.max_y
    out = r.matches
    for p in places:
        if x0 <= p.lng <= x1 and y0 <= p.lat <= y1 and (category is None or p.category == category):
            out.append(p)
    r.comparisons = len(places)
    r.nanos = perf_counter_ns() - t0
    return r


# Time complexity: O(n log n)
# Justification: building the filtered candidate list is O(n); sorting up to n candidates
# with Python's Timsort is O(n log n); slicing the first k is O(k). The sort dominates.
# (A heap-based top-k would be O(n log k), but this function is only a test baseline.)
def nearest_linear(places: List[Place], lat: float, lng: float, k: int,
                   category: Optional[str] = None) -> List[Place]:
    """Baseline for correctness testing of the quadtree nearest-k search."""
    cand = [p for p in places if category is None or p.category == category]
    cand.sort(key=lambda p: (p.lat - lat) ** 2 + (p.lng - lng) ** 2)
    return cand[:k]


# Time complexity: O(n)
# Justification: one pass over all n places; each place needs a category check and one
# O(1) distance_km call. No sorting and no early exit.
def radius_linear(places: List[Place], lat: float, lng: float, radius_km: float,
                  category: Optional[str] = None) -> List[Place]:
    """Baseline for correctness testing of the quadtree radius search."""
    return [p for p in places if (category is None or p.category == category)
            and distance_km(lat, lng, p.lat, p.lng) <= radius_km]


# Time complexity: O(k log k), k = larger of the two result sizes
# Justification: building each id list is O(k); sorting each list is O(k log k); comparing
# two sorted lists is O(k). The two sorts dominate.
def same_ids(a: SearchResult, b: SearchResult) -> bool:
    """True if both results contain exactly the same places."""
    return sorted(p.id for p in a.matches) == sorted(p.id for p in b.matches)
