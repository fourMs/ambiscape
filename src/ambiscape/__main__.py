"""``python -m ambiscape``: the same command line as ``ambiscape``."""
import sys

from .cli import main

sys.exit(main())
