"""Build only the lab feature caches that do not exist yet. python -m possession.featcache_missing G1 G1r1 ..."""
import os, sys
from possession.featcache import get, DIR
for n in sys.argv[1:]:
    if not os.path.exists(os.path.join(DIR, f"feat_{n}.npz")):
        print("building", n, flush=True); get(n)
print("lab features ready")
