"""Generate the final wing planform figure."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._final_geometry_figures import draw_planform


def main(argv=None):
    draw_planform(argv)


if __name__ == "__main__":
    main()
