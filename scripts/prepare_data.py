import argparse
from assets import ensure_structure, add_local_asset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="+", default=["cat"])
    ap.add_argument("--add", nargs="*")
    args = ap.parse_args()
    ensure_structure(args.species)
    if args.add:
        for pair in args.add:
            if "=" in pair:
                sp, path = pair.split("=", 1)
                add_local_asset(path, sp)
if __name__ == "__main__":
    main()
