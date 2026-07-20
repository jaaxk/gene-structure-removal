#!/usr/bin/env python3
"""Atomically pop the first line off a JSONL queue file (flock-protected) and
print it to stdout. Prints nothing (exit 0) if the queue is empty/missing."""
import fcntl
import sys

path = sys.argv[1]

with open(path, "a+") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.seek(0)
    lines = fh.readlines()
    if not lines:
        sys.exit(0)
    first, rest = lines[0], lines[1:]
    fh.seek(0)
    fh.truncate()
    fh.writelines(rest)
    fh.flush()
print(first.rstrip("\n"))
