"""
main.py  --  DRIVER for the Smart Location Finder (Quadtree edition)

Run from the project folder:

    python main.py            -> GUI if a display exists, else console demo
    python main.py gui        -> map window (drag = range query, click = nearest 5)
    python main.py demo       -> interactive console demo
    python main.py turtle     -> turtle drawing (optional: turtle 5000)
    python main.py bench      -> full benchmark, writes results_quadtree.csv
    python main.py bench 1000 -> same, with 1000 queries per experiment
    python main.py test       -> randomised self-test of every operation

Project layout (one file per team member)
    models.py      Member 1  data layer, SearchResult, linear-scan baselines
    quadtree.py    Member 2  QuadNode, QuadTree (insert / delete / invariants)
    queries.py     Member 3  range, radius, nearest-k, benchmark, self-test
    visualizer.py  Member 4  MapGUI, turtle demo, console demo
    main.py        Driver    SmartLocationFinder, wires everything together

Complexity notation used in the comments above each function:
    n = number of places stored / supplied      k = number of results returned
    N = number of quadtree nodes (about n/c for evenly spread data; measured ~0.37n at n = 100,000)
    c = leaf capacity (default 8)               D = max_depth (default 20)
    d = depth of the leaf reached (d <= D; about log4(n/c) when places are evenly spread)
    m = number of queries / repetitions
"""
from __future__ import annotations

import sys
from typing import List

from queries import run_benchmark, run_self_test
from visualizer import MapGUI, console_demo, turtle_demo


class SmartLocationFinder:
    """Driver class: reads the command line and starts the requested mode."""

    USAGE = "Usage: python main.py [gui | demo | turtle [n] | bench [queries] | test]"

    # Time complexity: O(1)
    # Justification: stores the argument list and reads one element from it.
    def __init__(self, argv: List[str]) -> None:
        self.argv = argv
        self.mode = argv[1].lower() if len(argv) > 1 else None

    # Time complexity: O(1)
    # Justification: one length check and one string-to-int conversion of a short argument.
    def _int_arg(self, default: int) -> int:
        return int(self.argv[2]) if len(self.argv) > 2 else default

    # Time complexity: same as run_self_test, O(q * n log n) with fixed sizes (see queries.py).
    # Justification: prints two lines and calls run_self_test(); nothing else.
    def run_tests(self) -> int:
        print("Running self-test...")
        passed = run_self_test()
        print("ALL TESTS PASSED" if passed else "SOME TESTS FAILED")
        return 0 if passed else 1

    # Time complexity: O(n log n) for the default 10,000 places (building the index), then the
    #   GUI event loop. If no display exists, falls back to console_demo (O(n log n) start-up).
    # Justification: MapGUI() builds the quadtree once; the try / except adds only O(1).
    def run_gui(self) -> int:
        try:
            MapGUI().run()
        except Exception as ex:                       # no tkinter or no display
            if self.mode == "gui":
                print(f"Could not open the GUI: {ex}")
                return 1
            print(f"(No GUI available: {ex}) Starting console demo.")
            console_demo()
        return 0

    # Time complexity: O(1) to dispatch + the cost of the chosen mode:
    #   bench   -> run_benchmark        test -> run_self_test
    #   demo    -> console_demo         turtle -> turtle_demo
    #   gui     -> run_gui              (anything else prints the usage text)
    # Justification: an if / elif chain with at most 6 string comparisons, which is a constant
    # number of steps; all real work happens in the function it calls.
    def run(self) -> int:
        if self.mode == "bench":
            run_benchmark(queries=self._int_arg(300))
        elif self.mode == "test":
            return self.run_tests()
        elif self.mode == "demo":
            console_demo()
        elif self.mode == "turtle":
            turtle_demo(self._int_arg(2000))
        elif self.mode in (None, "gui"):
            return self.run_gui()
        else:
            print(self.USAGE)
        return 0


if __name__ == "__main__":
    sys.exit(SmartLocationFinder(sys.argv).run())
