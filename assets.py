import argparse, os, shutil
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "animals"
def ensure_structure(species_list):
    for sp in species_list:
        p = DATA_DIR / sp
        p.mkdir(parents=True, exist_ok=True)
        print(f"Ensured: {p.resolve()}")
def add_local_asset(src_path, species_name, filename=None):
    species_dir = DATA_DIR / species_name
    species_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = os.path.basename(src_path)
    dst = species_dir / filename
    shutil.copy(src_path, dst)
    print(f"Added: {dst.resolve()}")
def list_assets(full_paths=True):
    assets = {}
    if DATA_DIR.exists():
        for sp in sorted([d for d in DATA_DIR.iterdir() if d.is_dir()]):
            files = list(sp.glob("*.wav"))
            assets[sp.name] = [str(f.resolve()) if full_paths else f.name for f in files]
    return assets
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensure", nargs="*")
    ap.add_argument("--add", type=str)
    ap.add_argument("--species", type=str)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.ensure: ensure_structure(args.ensure)
    if args.add and args.species: add_local_asset(args.add, args.species, args.name)
    if args.list: print(list_assets(full_paths=True))
    if not any([args.ensure, args.add, args.list]): print("No args. Try --help.")
if __name__ == "__main__":
    main()
