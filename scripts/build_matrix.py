import argparse

from milan.etl.build_matrix import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild all days from scratch")
    build(force=parser.parse_args().force)