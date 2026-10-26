"""
visualizer.py  --  MEMBER 4: Interactive map and demo interfaces

Contains:
    * range_report   (runs a range query on quadtree + linear scan and formats the comparison)
    * console_demo   (text-command interface)
    * MapGUI         (Tkinter map: drag = range query, click = 5 nearest)
    * turtle_demo    (turtle drawing of the quadtree and the places)

Complexity notation used in the comments above each function:
    n = number of places stored / supplied      k = number of results returned
    N = number of quadtree nodes (about n/c for evenly spread data; measured ~0.37n at n = 100,000)
    c = leaf capacity (default 8)               D = max_depth (default 20)
    d = depth of the leaf reached (d <= D; about log4(n/c) when places are evenly spread)
    m = number of queries / repetitions
"""
from __future__ import annotations

from time import perf_counter_ns
from typing import List, Optional

from models import (CATEGORIES, LAT_MAX, LAT_MIN, LNG_MAX, LNG_MIN, Place, Rect, distance_km,
                    generate_places, linear_search, same_ids)
from quadtree import QuadTree, build_tree
from queries import nearest, radius_search, range_search


# ==============================================================================
# Shared helper
# ==============================================================================

# Time complexity: O(reps * (R + n) + k log k), R = cost of range_search (see queries.py).
# Justification: for each of the reps repetitions it runs the quadtree search (R) and the
# linear scan (O(n)). same_ids sorts both result lists once: O(k log k). Formatting only
# the first 10 matches is O(1). With reps = 1 (console / turtle) this is O(R + n + k log k).
def range_report(tree: QuadTree, places: List[Place], q: Rect, cat: Optional[str], reps: int = 1):
    """Run the range query on both methods; returns (quadtree result, linear result, text)."""
    a = b = None
    qn = ln = 0
    for _ in range(reps):
        a = range_search(tree, q, cat)
        b = linear_search(places, q, cat)
        qn += a.nanos
        ln += b.nanos
    qn, ln = qn / reps / 1000.0, ln / reps / 1000.0
    lines = [f"Matches: {len(a.matches)}",
             f"Quadtree: {a.comparisons:,} comparisons, {a.nodes_visited:,} nodes, {qn:.1f} us",
             f"Linear:   {b.comparisons:,} comparisons, {ln:.1f} us",
             f"Speedup: {ln / qn:.1f}x   same result: {'yes' if same_ids(a, b) else 'NO'}", ""]
    for p in a.matches[:10]:
        lines.append(f"  {p}")
    if len(a.matches) > 10:
        lines.append(f"  ... {len(a.matches) - 10} more")
    return a, b, "\n".join(lines)


# ==============================================================================
# Console demo
# ==============================================================================

# Time complexity: O(n log n) start-up, then per command:
#   range O(R + n + k log k) | near O(nearest) | radius O(R) | get O(1)
#   del O(n) | stats O(N + n * d) | quit O(1)
# Justification: start-up generates n = 10,000 places (O(n)) and builds the tree
# (O(n log n)). Each command does one tree query or lookup. 'get' is a dict lookup, O(1).
# 'del' is O(n) because places.remove(p) scans the Python list, even though tree.delete is
# only O(c * d). The loop itself waits for input, so its cost is the sum of the commands.
def console_demo() -> None:
    n = 10_000
    places = generate_places(n, 42)
    by_id = {p.id: p for p in places}
    tree = build_tree(places)
    print(f"Built quadtree: {tree.size():,} places, height {tree.height()}, {tree.node_count():,} nodes, "
          f"invariants: {'OK' if tree.check_invariants() is None else 'BROKEN'}")
    print(f"Map bounds: lat {LAT_MIN}-{LAT_MAX}, lng {LNG_MIN}-{LNG_MAX}")
    print("Categories: " + ", ".join(CATEGORIES))
    print("Commands:")
    print("  range minLat minLng maxLat maxLng [category]")
    print("  near  lat lng k [category]")
    print("  radius lat lng km [category]")
    print("  get id        |  del id        |  stats        |  quit")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "quit":
            break
        try:
            t = line.split(None, 5)
            cmd = t[0]
            if cmd == "range":
                la1, ln1, la2, ln2 = map(float, t[1:5])
                cat = t[5] if len(t) > 5 else None
                q = Rect(min(ln1, ln2), min(la1, la2), max(ln1, ln2), max(la1, la2))
                print(range_report(tree, places, q, cat)[2])
            elif cmd == "near":
                lat, lng, k = float(t[1]), float(t[2]), int(t[3])
                cat = " ".join(t[4:]) if len(t) > 4 else None
                for p in nearest(tree, lat, lng, k, cat):
                    print(f"  {p}")
            elif cmd == "radius":
                lat, lng, km = float(t[1]), float(t[2]), float(t[3])
                cat = " ".join(t[4:]) if len(t) > 4 else None
                r = radius_search(tree, lat, lng, km, cat)
                print(f"Matches: {len(r.matches)} ({r.comparisons:,} comparisons, {r.nodes_visited:,} nodes)")
                for p in r.matches[:10]:
                    print(f"  {p}")
            elif cmd == "get":
                p = by_id.get(int(t[1]))
                print(f"  {p}" if p else "  not found")
            elif cmd == "del":
                p = by_id.get(int(t[1]))
                if p and tree.delete(p):
                    del by_id[p.id]
                    places.remove(p)
                    print(f"  deleted {p}")
                else:
                    print("  not found")
            elif cmd == "stats":
                print(f"  places {tree.size():,}, height {tree.height()}, nodes {tree.node_count():,}, "
                      f"capacity {tree.capacity}, invariants "
                      f"{'OK' if tree.check_invariants() is None else 'BROKEN'}")
            else:
                print("Unknown command.")
        except (ValueError, IndexError) as ex:
            print(f"Bad input: {ex}")


# ==============================================================================
# Tkinter map GUI
# ==============================================================================

class MapGUI:
    """Tkinter demo. Drag = range query (quadtree vs linear), click = 5 nearest of chosen category."""
    W, H, PAD = 860, 720, 10
    REPS = 20            # repeat each query to get a stable average time
    MAX_DRAW = 30000     # draw at most this many background points (tkinter gets slow beyond this)
    COLORS = ["#D32F2F", "#F57C00", "#388E3C", "#7B1FA2", "#00796B", "#1976D2"]

    # Time complexity: O(n log n) for the default n = 10,000 (dominated by rebuild()).
    # Justification: creating the Tkinter widgets is a constant number of O(1) calls; the
    # constructor then calls rebuild(), which generates and indexes the places.
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Smart Location Finder - Quadtree demo")
        self.places: List[Place] = []
        self.tree = QuadTree()
        self.start = None

        top = ttk.Frame(self.root)
        top.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(top, text="Places:").pack(side="left")
        self.size_box = ttk.Combobox(top, width=8, state="readonly",
                                     values=[1000, 10000, 50000, 100000])
        self.size_box.set(10000)
        self.size_box.pack(side="left", padx=4)
        ttk.Button(top, text="Build index", command=self.rebuild).pack(side="left")
        ttk.Label(top, text="   Category:").pack(side="left")
        self.cat_box = ttk.Combobox(top, width=12, state="readonly", values=["All"] + CATEGORIES)
        self.cat_box.set("All")
        self.cat_box.pack(side="left", padx=4)
        self.show_index = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Show quadtree leaf boxes", variable=self.show_index,
                        command=self.redraw).pack(side="left", padx=8)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, width=self.W, height=self.H, bg="white", highlightthickness=0)
        self.canvas.pack(side="left")
        self.info = tk.Text(body, width=52, font=("Courier", 10), state="disabled")
        self.info.pack(side="left", fill="y")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.rebuild()

    # ---- coordinate mapping
    # Time complexity: O(1)
    # Justification: two linear interpolations (a few multiplications and additions).
    def to_px(self, lng: float, lat: float):
        x = self.PAD + (lng - LNG_MIN) / (LNG_MAX - LNG_MIN) * (self.W - 2 * self.PAD)
        y = self.H - self.PAD - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (self.H - 2 * self.PAD)
        return x, y

    # Time complexity: O(1)
    # Justification: the inverse of to_px, again a fixed number of arithmetic operations.
    def to_geo(self, x: float, y: float):
        lng = LNG_MIN + (x - self.PAD) / (self.W - 2 * self.PAD) * (LNG_MAX - LNG_MIN)
        lat = LAT_MIN + (self.H - self.PAD - y) / (self.H - 2 * self.PAD) * (LAT_MAX - LAT_MIN)
        return lng, lat

    # Time complexity: O(1)
    # Justification: reads the combobox value and compares it with 'All'.
    def category(self) -> Optional[str]:
        c = self.cat_box.get()
        return None if c == "All" else c

    # Time complexity: O(|CATEGORIES|) = O(1)
    # Justification: list.index scans at most the 6 category names; the list has a fixed size.
    def color(self, p: Place) -> str:
        return self.COLORS[CATEGORIES.index(p.category) % len(self.COLORS)]

    # Time complexity: O(L), L = length of the text shown (bounded by a few hundred characters).
    # Justification: Tk clears and re-inserts the text once; no loops over places.
    def set_info(self, text: str) -> None:
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("end", text)
        self.info.config(state="disabled")

    # ---- actions
    # Time complexity: O(n log n)
    # Justification: generate_places is O(n); build_tree is O(n log n) on average;
    # check_invariants is O(N + n * d); redraw() is O(N + min(n, 30000)). The tree
    # construction dominates.
    def rebuild(self) -> None:
        n = int(self.size_box.get())
        self.places = generate_places(n, 42)
        t0 = perf_counter_ns()
        self.tree = build_tree(self.places)
        ms = (perf_counter_ns() - t0) / 1e6
        self.redraw()
        self.set_info(f"Built quadtree: {n:,} places in {ms:.0f} ms\n"
                      f"Height {self.tree.height()}, {self.tree.node_count():,} nodes\n"
                      f"Invariants: {'OK' if self.tree.check_invariants() is None else 'BROKEN'}\n\n"
                      f"Drag on the map  -> range query\nClick on the map -> 5 nearest")

    # Time complexity: O(N + min(n, MAX_DRAW)) with MAX_DRAW = 30,000
    # Justification: when 'Show quadtree leaf boxes' is on, leaf_rects() visits all N nodes and
    # one rectangle is drawn per non-empty leaf. Then at most MAX_DRAW places are drawn: the
    # list is sampled with a step of n / MAX_DRAW, and each draw is O(1).
    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self.show_index.get():
            for r in self.tree.leaf_rects():
                x0, y1 = self.to_px(r.min_x, r.min_y)
                x1, y0 = self.to_px(r.max_x, r.max_y)
                c.create_rectangle(x0, y0, x1, y1, outline="#B0BEC5", tags="idx")
        step = max(1, len(self.places) // self.MAX_DRAW)
        for p in self.places[::step]:
            x, y = self.to_px(p.lng, p.lat)
            c.create_rectangle(x, y, x + 1.5, y + 1.5, fill=self.color(p), outline="", tags="pts")
        if step > 1:
            c.create_text(8, 8, anchor="nw", fill="#666",
                          text=f"(showing every {step}th of {len(self.places):,} points)")

    # Time complexity: O(1)
    # Justification: stores the start point and deletes the single old selection rectangle.
    def on_press(self, e) -> None:
        self.start = (e.x, e.y)
        self.canvas.delete("q")

    # Time complexity: O(1)
    # Justification: deletes the old selection rectangle and draws one new one.
    def on_drag(self, e) -> None:
        self.canvas.delete("q")
        self.canvas.create_rectangle(*self.start, e.x, e.y, outline="#1976D2", width=2, tags="q")

    # Time complexity: O(H) + cost of the query it starts, H <= 3,000 previously drawn hits.
    # Justification: deleting the 'hit' canvas items is linear in the number of items drawn by
    # the last query (at most 3,000 for a range query). It then dispatches to do_nearest or
    # do_range based on one O(1) distance check.
    def on_release(self, e) -> None:
        self.canvas.delete("hit")
        sx, sy = self.start
        if abs(e.x - sx) < 4 and abs(e.y - sy) < 4:
            self.do_nearest(e.x, e.y)
        else:
            self.do_range(sx, sy, e.x, e.y)

    # Time complexity: O(REPS * (R + n) + min(k, 3000)), REPS = 20 repetitions.
    # Justification: range_report repeats the quadtree search (R) and the linear scan (O(n))
    # REPS times to get a stable timing; drawing the highlighted matches costs O(1) each for
    # at most 3,000 of them.
    def do_range(self, x0, y0, x1, y1) -> None:
        lng_a, lat_a = self.to_geo(x0, y0)
        lng_b, lat_b = self.to_geo(x1, y1)
        q = Rect(min(lng_a, lng_b), min(lat_a, lat_b), max(lng_a, lng_b), max(lat_a, lat_b))
        a, _, text = range_report(self.tree, self.places, q, self.category(), self.REPS)
        for p in a.matches[:3000]:
            x, y = self.to_px(p.lng, p.lat)
            self.canvas.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill="#FFEB3B",
                                    outline="#F57F17", tags="hit")
        self.set_info(f"RANGE QUERY (category: {self.cat_box.get()})\n" + text)

    # Time complexity: O(nearest) + O(k), k = 5.
    # Justification: one best-first nearest search (see queries.nearest), then a constant
    # number (5) of O(1) distance and canvas-drawing operations.
    def do_nearest(self, px, py) -> None:
        lng, lat = self.to_geo(px, py)
        res = nearest(self.tree, lat, lng, 5, self.category())
        self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#1976D2", outline="", tags="hit")
        lines = [f"5 NEAREST to ({lat:.5f}, {lng:.5f})  category: {self.cat_box.get()}", ""]
        for i, p in enumerate(res, 1):
            x, y = self.to_px(p.lng, p.lat)
            self.canvas.create_line(px, py, x, y, fill="#1976D2", dash=(3, 3), tags="hit")
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#FFEB3B", outline="#F57F17",
                                    width=2, tags="hit")
            self.canvas.create_text(x + 8, y - 8, text=str(i), anchor="w", tags="hit",
                                    font=("Arial", 10, "bold"))
            lines.append(f"{i}. {p}  ({distance_km(lat, lng, p.lat, p.lng):.2f} km)")
        self.set_info("\n".join(lines))

    # Time complexity: O(1) to start; the Tk main loop then handles one event at a time.
    # Justification: mainloop() just waits for events; each event costs as described on the
    # handler above.
    def run(self) -> None:
        self.root.mainloop()


# ==============================================================================
# Turtle demo
# ==============================================================================

# Time complexity: O(n log n + N + n) set-up = O(n log n), then per click as below.
# Justification: generate_places O(n) and build_tree O(n log n); drawing the leaf boxes is
# O(N) and drawing every place is O(n), each draw being O(1). Per-event costs are given on
# the nested handlers. (turtle draws slowly in practice, so keep n small, e.g. 2,000.)
def turtle_demo(n: int = 2000) -> None:
    """Draw the quadtree and the places with the turtle module.

    Click = 5 nearest places.   Key r = range mode (click two opposite corners).
    Key n = nearest mode.       Key c = cycle category filter.   Key q = quit.
    Full numbers are printed in the terminal; the window title shows a summary.
    """
    import turtle

    places = generate_places(n, 42)
    tree = build_tree(places)
    colors = MapGUI.COLORS
    pad_x, pad_y = 0.02, 0.05

    screen = turtle.Screen()
    screen.setup(900, 800)
    screen.title(f"Quadtree - {n:,} places, height {tree.height()}, {tree.node_count():,} nodes")
    screen.tracer(0)                       # draw everything first, then show it in one go
    screen.setworldcoordinates(LNG_MIN - pad_x, LAT_MIN - pad_y, LNG_MAX + pad_x, LAT_MAX + pad_y)

    # Time complexity: O(1)
    # Justification: creates one Turtle object and sets four properties.
    def new_turtle():
        t = turtle.Turtle(visible=False)
        t.hideturtle()
        t.penup()
        t.speed(0)
        return t

    base, over, label = new_turtle(), new_turtle(), new_turtle()

    # Time complexity: O(1)
    # Justification: moves the pen to 4 corners; a fixed number of moves.
    def draw_rect(t, r: Rect, color: str, width: int) -> None:
        t.color(color)
        t.pensize(width)
        t.penup()
        t.goto(r.min_x, r.min_y)
        t.pendown()
        for x, y in ((r.max_x, r.min_y), (r.max_x, r.max_y), (r.min_x, r.max_y), (r.min_x, r.min_y)):
            t.goto(x, y)
        t.penup()

    # static layer: quadtree leaf boxes + all places
    draw_rect(base, tree.bounds, "#546E7A", 2)
    for r in tree.leaf_rects():
        draw_rect(base, r, "#CFD8DC", 1)
    for p in places:
        base.goto(p.lng, p.lat)
        base.dot(4, colors[CATEGORIES.index(p.category) % len(colors)])

    state = {"mode": "nearest", "corner": None, "cat_i": 0}
    cats = [None] + CATEGORIES

    # Time complexity: O(1)
    # Justification: clears and rewrites one line of text.
    def set_label(text: str) -> None:
        label.clear()
        label.color("black")
        label.goto(LNG_MIN - pad_x + 0.004, LAT_MAX + 0.012)
        label.write(text, font=("Courier", 10, "normal"))

    # Time complexity: O(R + n + k log k + min(k, 3000)).
    # Justification: one range_report (quadtree search R + linear scan O(n) + same_ids sort),
    # then at most 3,000 O(1) dot drawings for the matches.
    def show_range(q: Rect) -> None:
        over.clear()
        a, _, text = range_report(tree, places, q, cats[state["cat_i"]])
        draw_rect(over, q, "#1976D2", 3)
        for p in a.matches[:3000]:
            over.goto(p.lng, p.lat)
            over.dot(8, "#FFEB3B")
            over.dot(4, "#F57F17")
        set_label(f"RANGE: {len(a.matches)} matches  (mode: {state['mode']}, category: {cats[state['cat_i']] or 'All'})")
        print("\nRANGE QUERY\n" + text)
        screen.update()

    # Time complexity: O(nearest) + O(k), k = 5.
    # Justification: one best-first nearest search, then a loop over the 5 results, each doing
    # a fixed number of drawing / print operations.
    def show_nearest(x: float, y: float) -> None:
        over.clear()
        cat = cats[state["cat_i"]]
        res = nearest(tree, y, x, 5, cat)
        print(f"\n5 NEAREST to ({y:.5f}, {x:.5f}), category: {cat or 'All'}")
        over.color("#1976D2")
        over.goto(x, y)
        over.dot(10, "#1976D2")
        for i, p in enumerate(res, 1):
            over.color("#1976D2")
            over.pensize(1)
            over.goto(x, y)
            over.pendown()
            over.goto(p.lng, p.lat)
            over.penup()
            over.dot(12, "#FFEB3B")
            over.dot(6, "#F57F17")
            over.color("black")
            over.goto(p.lng + 0.003, p.lat + 0.003)
            over.write(str(i), font=("Arial", 11, "bold"))
            print(f"  {i}. {p}  ({distance_km(y, x, p.lat, p.lng):.2f} km)")
        set_label(f"NEAREST 5  (mode: {state['mode']}, category: {cat or 'All'})")
        screen.update()

    # Time complexity: O(1) to dispatch, plus the cost of show_nearest or show_range.
    # Justification: a few dictionary lookups and a branch; the heavy work is in the handler
    # it calls.
    def on_click(x: float, y: float) -> None:
        if state["mode"] == "nearest":
            show_nearest(x, y)
        elif state["corner"] is None:
            state["corner"] = (x, y)
            over.goto(x, y)
            over.dot(8, "#1976D2")
            screen.update()
        else:
            (x0, y0), state["corner"] = state["corner"], None
            show_range(Rect(min(x0, x), min(y0, y), max(x0, x), max(y0, y)))

    # Time complexity: O(1)
    # Justification: updates two dictionary entries and rewrites one label.
    def set_mode(mode: str) -> None:
        state["mode"], state["corner"] = mode, None
        set_label(f"mode: {mode}   (r = range, n = nearest, c = category, q = quit)")
        screen.update()

    # Time complexity: O(1)
    # Justification: increments an index modulo 7 and rewrites one label.
    def cycle_category() -> None:
        state["cat_i"] = (state["cat_i"] + 1) % len(cats)
        set_label(f"category: {cats[state['cat_i']] or 'All'}   (mode: {state['mode']})")
        screen.update()

    screen.onclick(on_click)
    screen.onkey(lambda: set_mode("range"), "r")
    screen.onkey(lambda: set_mode("nearest"), "n")
    screen.onkey(cycle_category, "c")
    screen.onkey(screen.bye, "q")
    screen.listen()

    # start with a sample range query so the window is not empty
    show_range(Rect(80.15, 12.95, 80.28, 13.10))
    print("\nClick for 5 nearest. Keys: r = range mode, n = nearest mode, c = category, q = quit.")
    screen.mainloop()
