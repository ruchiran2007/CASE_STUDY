"""
quadtree.py  --  MEMBER 2: Quadtree index (the data structure itself)

Contains:
    * QuadNode   (one square region of the map)
    * QuadTree   (insert, split, delete with merge, introspection, invariant checker)
    * build_tree (helper that inserts a list of places)

Design notes
    * Points live only in leaves. A leaf splits into 4 equal quadrants when it
      holds more than `capacity` points.
    * A point is routed to exactly ONE child (x >= mid -> east, y >= mid -> north),
      so a point on a shared edge is never stored twice.
    * max_depth stops infinite subdivision when many places share one coordinate;
      a leaf at max_depth is allowed to hold more than `capacity` points.
    * Delete merges four sibling leaves back into their parent once they fit in
      one leaf again, so the tree shrinks.

The search algorithms (range, radius, nearest-k) live in queries.py.

Complexity notation used in the comments above each function:
    n = number of places stored / supplied      k = number of results returned
    N = number of quadtree nodes (about n/c for evenly spread data; measured ~0.37n at n = 100,000)
    c = leaf capacity (default 8)               D = max_depth (default 20)
    d = depth of the leaf reached (d <= D; about log4(n/c) when places are evenly spread)
    m = number of queries / repetitions
"""
from __future__ import annotations

from typing import List, Optional

from models import LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, Place, Rect


class QuadNode:
    __slots__ = ("bounds", "depth", "mx", "my", "places", "children")

    # Time complexity: O(1)
    # Justification: computes two midpoints and creates two empty containers; the node never
    # looks at any place when it is created.
    def __init__(self, bounds: Rect, depth: int) -> None:
        self.bounds = bounds
        self.depth = depth
        self.mx = (bounds.min_x + bounds.max_x) / 2.0
        self.my = (bounds.min_y + bounds.max_y) / 2.0
        self.places: List[Place] = []                   # used by leaves only
        self.children: Optional[List[QuadNode]] = None  # [SW, SE, NW, NE] or None for a leaf

    # Time complexity: O(1)
    # Justification: two comparisons and a bitwise OR choose one of the 4 quadrants, so the
    # tree can descend one level in constant time.
    def child_index(self, x: float, y: float) -> int:
        return (1 if x >= self.mx else 0) | (2 if y >= self.my else 0)


class QuadTree:
    """Point-region quadtree over places (x = lng, y = lat)."""

    # Time complexity: O(1)
    # Justification: stores the settings and creates a single empty root node.
    def __init__(self, bounds: Optional[Rect] = None, capacity: int = 8, max_depth: int = 20) -> None:
        self.bounds = bounds or Rect(LNG_MIN, LAT_MIN, LNG_MAX, LAT_MAX)
        self.capacity = capacity
        self.max_depth = max_depth
        self.root = QuadNode(self.bounds, 1)
        self._size = 0

    # Time complexity: O(1)
    # Justification: returns a counter that insert() and delete() keep up to date, so there is
    # no need to traverse the tree.
    def size(self) -> int:
        return self._size

    # ------------------------------------------------------------------ insert
    # Time complexity: O(d) per insert, d = depth of the leaf reached (d <= D).
    #   Evenly spread data: d ~ log4(n/c), so O(log n).
    #   Worst case (heavily clustered / coincident points): O(D * c); D and c are fixed
    #   constants (20 and 8), so the cost is still bounded independently of n.
    # Justification: the while loop moves down exactly one level per iteration using
    # child_index (O(1)). The only extra work is _split (O(c)), which runs only when the leaf
    # is full; a chain of splits can happen at most D times. Appending to a leaf list is
    # amortised O(1). Inserting n places therefore costs O(n log n) on average.
    def insert(self, p: Place) -> bool:
        """O(depth) per insert. Returns False if the place lies outside the map bounds."""
        if not self.bounds.contains_point(p.lng, p.lat):
            return False
        node = self.root
        while True:
            if node.children is None:
                if len(node.places) < self.capacity or node.depth >= self.max_depth:
                    node.places.append(p)
                    self._size += 1
                    return True
                self._split(node)
            node = node.children[node.child_index(p.lng, p.lat)]

    # Time complexity: O(c)
    # Justification: creating the 4 child nodes is O(1); moving the c places of the full leaf
    # into the right child costs O(1) per place (child_index + append), so c * O(1) = O(c).
    # c is a small constant (default 8).
    def _split(self, node: QuadNode) -> None:
        b, mx, my, d = node.bounds, node.mx, node.my, node.depth + 1
        node.children = [
            QuadNode(Rect(b.min_x, b.min_y, mx, my), d),      # SW
            QuadNode(Rect(mx, b.min_y, b.max_x, my), d),      # SE
            QuadNode(Rect(b.min_x, my, mx, b.max_y), d),      # NW
            QuadNode(Rect(mx, my, b.max_x, b.max_y), d),      # NE
        ]
        old, node.places = node.places, []
        for p in old:
            node.children[node.child_index(p.lng, p.lat)].places.append(p)

    # ------------------------------------------------------------------ delete
    # Time complexity: O(c * d) worst case; O(log n) on average for evenly spread data.
    # Justification: (1) descending to the leaf is O(d); (2) scanning that leaf's list for the
    # id is O(c) (O(m) if m coincident places pile up in a leaf at max_depth); (3) the merge
    # loop walks back up at most d levels, doing O(1) work per level to test the 4 children and
    # only rebuilding the parent's list (<= c places, O(c)) when a merge really happens.
    def delete(self, p: Place) -> bool:
        """Remove the place with p.id (located via p's coordinates), then merge sibling leaves."""
        if not self.bounds.contains_point(p.lng, p.lat):
            return False
        path: List[QuadNode] = []
        node = self.root
        while node.children is not None:
            path.append(node)
            node = node.children[node.child_index(p.lng, p.lat)]
        for i, q in enumerate(node.places):
            if q.id == p.id:
                node.places.pop(i)
                break
        else:
            return False
        self._size -= 1
        for parent in reversed(path):                      # collapse upwards while possible
            kids = parent.children
            if any(c.children is not None for c in kids):
                break
            if sum(len(c.places) for c in kids) > self.capacity:
                break
            parent.places = [q for c in kids for q in c.places]
            parent.children = None
        return True

    # ------------------------------------------------------------ introspection
    # Time complexity: O(N) to iterate through the whole generator
    # Justification: every node is pushed on the stack once and popped once, and each pop does
    # O(1) work plus pushing its (at most 4) children. N is about n/c for evenly spread data.
    def _nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            if n.children is not None:
                stack.extend(n.children)

    # Time complexity: O(N)
    # Justification: must look at every node's depth to find the maximum, so it visits all N
    # nodes once (via _nodes) and does O(1) work on each.
    def height(self) -> int:
        return max(n.depth for n in self._nodes())

    # Time complexity: O(N)
    # Justification: counts the nodes produced by _nodes, which visits each of the N nodes
    # exactly once with O(1) work per node.
    def node_count(self) -> int:
        return sum(1 for _ in self._nodes())

    # Time complexity: O(N)
    # Justification: one pass over all N nodes (via _nodes); each node needs an O(1) leaf /
    # non-empty check and at most one append.
    def leaf_rects(self) -> List[Rect]:
        """Bounds of all non-empty leaves (used by the visualizer to draw the index)."""
        return [n.bounds for n in self._nodes() if n.children is None and n.places]

    # Time complexity: O(N + n)
    # Justification: visits all N nodes once, and for each leaf copies its places into the
    # result; across all leaves that copies every one of the n places exactly once.
    def all_places(self) -> List[Place]:
        return [p for n in self._nodes() if n.children is None for p in n.places]

    # Time complexity: O(d), d <= D
    # Justification: follows one child pointer per level using child_index (O(1)) from the
    # root to a leaf, so it takes as many steps as the depth of that leaf.
    def _locate(self, p: Place) -> QuadNode:
        node = self.root
        while node.children is not None:
            node = node.children[node.child_index(p.lng, p.lat)]
        return node

    # Time complexity: O(N + n * d)
    # Justification: visiting the N nodes costs O(N) with O(1) checks each. For every one of
    # the n stored places it also calls _locate, which costs O(d), giving O(n * d). It is a
    # debugging / testing aid and is not used by the search algorithms.
    def check_invariants(self) -> Optional[str]:
        """Return None if the tree is valid, else a message describing the problem."""
        count = 0
        for n in self._nodes():
            if n.children is None:
                if len(n.places) > self.capacity and n.depth < self.max_depth:
                    return f"overfull leaf ({len(n.places)} places at depth {n.depth})"
                for p in n.places:
                    if not n.bounds.contains_point(p.lng, p.lat):
                        return "place outside its leaf"
                    if self._locate(p) is not n:
                        return "place stored in the wrong leaf"
                count += len(n.places)
            else:
                if n.places:
                    return "internal node holds places"
                for c in n.children:
                    if c.depth != n.depth + 1:
                        return "bad child depth"
                    if not n.bounds.contains(c.bounds):
                        return "child outside parent"
        if count != self._size:
            return f"entry count {count} != size {self._size}"
        return None


# Time complexity: O(n log n) on average; O(n * D) worst case.
# Justification: it performs n calls to insert(), each O(d) with d ~ log4(n/c) for evenly
# spread data (d <= D in the worst case). n * O(log n) = O(n log n).
def build_tree(places: List[Place], capacity: int = 8) -> QuadTree:
    """Create a tree and insert every place into it."""
    tree = QuadTree(capacity=capacity)
    for p in places:
        tree.insert(p)
    return tree
