"""Generate the final wing geometry figure."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._final_geometry_figures import draw_3d


def main(argv=None):
    draw_3d(False, argv)


if __name__ == "__main__":
    main()
