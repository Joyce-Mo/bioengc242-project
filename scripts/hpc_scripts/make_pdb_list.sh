#!/bin/bash
# Generate a text file listing all PDB paths (one per line).
# This file is used by job arrays to map task IDs to PDB files.
#
# Usage: bash make_pdb_list.sh /path/to/pdb_dir > pdb_list.txt

PDB_DIR="${1:?Usage: $0 /path/to/pdb_dir}"

find "$PDB_DIR" -name "*.pdb" -type f | sort
