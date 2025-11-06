#!/usr/bin/env python3
"""
move_fraction.py

Move a fraction of files from a source directory into a destination directory.

Default behavior: move 1/10 (10%) of files (non-recursive), at least 1 file when available.

Usage examples:
  python move_fraction.py /path/to/source /path/to/dest
  python move_fraction.py /src /dst --fraction 0.2 --seed 42 --pattern "*.jpg" --dry-run

"""

import os
import shutil
import random
import argparse
import fnmatch
import math
import logging
import sys


def collect_files(src_dir: str, pattern: str = None, recursive: bool = False):
    """Return a list of file paths (not directories) under src_dir.

    By default non-recursive (files directly in src_dir). If recursive True, walks subdirectories.
    If pattern is provided, it is applied to the basename using fnmatch.
    """
    files = []
    if recursive:
        for root, dirs, filenames in os.walk(src_dir):
            for name in filenames:
                if pattern is None or fnmatch.fnmatch(name, pattern):
                    files.append(os.path.join(root, name))
    else:
        try:
            for name in os.listdir(src_dir):
                path = os.path.join(src_dir, name)
                if os.path.isfile(path) and (pattern is None or fnmatch.fnmatch(name, pattern)):
                    files.append(path)
        except FileNotFoundError:
            raise
    return files


def move_sample(src_dir: str, dst_dir: str, fraction: float = 0.1, seed: int = None, pattern: str = None,
                recursive: bool = False, dry_run: bool = False, min_move: int = 1):
    """Move a fraction of files from src_dir to dst_dir.

    Returns list of moved (or would-be-moved when dry_run) file paths.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be > 0 and <= 1")

    files = collect_files(src_dir, pattern=pattern, recursive=recursive)
    logging.debug("Found %d candidate files", len(files))

    if not files:
        logging.info("No files found in %s matching pattern=%s", src_dir, pattern)
        return []

    if seed is not None:
        random.seed(seed)

    # Calculate number to move
    n_total = len(files)
    n_move = max(min_move, int(round(n_total * fraction)))
    # Don't exceed total
    n_move = min(n_move, n_total)

    chosen = random.sample(files, n_move)

    moved = []
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)

    for src_path in chosen:
        # Determine destination path — keep filename and preserve relative path if recursive and moved from subdir
        if recursive:
            # compute relative path from src_dir
            rel = os.path.relpath(src_path, start=src_dir)
            dst_path = os.path.join(dst_dir, rel)
            dst_parent = os.path.dirname(dst_path)
            if not dry_run:
                os.makedirs(dst_parent, exist_ok=True)
        else:
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        if dry_run:
            logging.info("Would move: %s -> %s", src_path, dst_path)
            moved.append((src_path, dst_path))
        else:
            logging.info("Moving: %s -> %s", src_path, dst_path)
            shutil.move(src_path, dst_path)
            moved.append((src_path, dst_path))

    return moved


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Move a fraction of files from source directory into destination directory.")
    p.add_argument("src", help="Source directory path")
    p.add_argument("dst", help="Destination directory path (will be created if missing)")
    p.add_argument("--fraction", type=float, default=0.1, help="Fraction of files to move (default 0.1)")
    p.add_argument("--seed", type=int, default=None, help="Random seed to make selection reproducible")
    p.add_argument("--pattern", default=None, help="Filename glob pattern (e.g. '*.jpg') to filter candidates")
    p.add_argument("--recursive", action="store_true", help="Include files in subdirectories and preserve relative paths")
    p.add_argument("--dry-run", action="store_true", help="Do not move files; just print what would be moved")
    p.add_argument("--min-move", type=int, default=1, help="Minimum number of files to move when there are any candidates (default 1)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    src = args.src
    dst = args.dst

    if not os.path.isdir(src):
        logging.error("Source directory does not exist or is not a directory: %s", src)
        sys.exit(2)

    # Prevent accidental overlapping source == destination
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)
    if abs_src == abs_dst:
        logging.error("Source and destination are the same: %s", src)
        sys.exit(2)

    try:
        moved = move_sample(src, dst, fraction=args.fraction, seed=args.seed, pattern=args.pattern,
                            recursive=args.recursive, dry_run=args.dry_run, min_move=args.min_move)
    except Exception as e:
        logging.exception("Failed to move files: %s", e)
        sys.exit(1)

    logging.info("Operation complete. %d files %s.", len(moved), "would be moved" if args.dry_run else "moved")


if __name__ == "__main__":
    main()
