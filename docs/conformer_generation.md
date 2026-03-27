# Synthetic Conformer Generation

This document describes the two methods used to generate protein conformational ensembles from single-structure PDB inputs: **Rosetta Backrub** and **MC-SCE** (Monte Carlo Side-Chain Ensemble).

Both pipelines are designed for HPC job arrays on Wynton (SGE) and Savio (SLURM), where each array task processes one PDB file.

## Overview

| Method | Script | What it samples | Score function | Reference |
|--------|--------|----------------|---------------|-----------|
| Backrub | `scripts/run_backrub.py` | Backbone + post-minimization | beta_nov16 (PyRosetta) | Smith & Kortemme (2008) |
| MC-SCE | `scripts/run_mcsce.py` | Side chains (fixed backbone) | AMBER ff14SB (MCSCE) + beta_nov16_cart post-minimization | THGLab/MCSCE |

---

## Shared Preprocessing

Both pipelines use the same preprocessing protocol on input structures (applied automatically for Backrub; opt-in via `--preprocess` for MC-SCE):

1. **Idealize bond geometries** — `IdealizeMover` adjusts bond lengths/angles to ideal Engh & Huber values while preserving backbone torsions (phi/psi/omega)
2. **Repack side chains** — `PackRotamersMover` with `RestrictToRepacking` + `IncludeCurrent`, scored by `beta_nov16`
3. **Enable flip_HNQ** — allows His/Asn/Gln sidechain flips during minimization
4. **Minimize (chi only)** — L-BFGS minimization of sidechain torsions only
5. **Minimize (chi + backbone)** — L-BFGS minimization of all torsions
6. **Disable flip_HNQ** — turned off for subsequent sampling

PyRosetta is initialized with `-corrections:beta_nov16` to enable the beta_nov16 energy function.

---

## Backrub Protocol

**Script:** `scripts/run_backrub.py`
**Submit script:** `scripts/submit_backrub_wynton.sh`

### Protocol steps

1. **Preprocessing** (idealize → repack → minimize, see above)
2. **Backrub MC sampling** — for each conformer:
   - Clone the preprocessed pose
   - Set up backbone backrub segments via `BackrubMover`
   - Run N Monte Carlo steps with Metropolis acceptance at temperature kT
   - Recover the lowest-energy pose from the trajectory
3. **Post-backrub minimization** — two-stage minimize (chi only, then chi + backbone) on each conformer
4. **Output** — one PDB per conformer + energy trajectory CSVs

### Default parameters

| Parameter | Default | Flag |
|-----------|---------|------|
| Conformers per structure | 5 | `--nconfs` |
| MC steps per conformer | 10,000 | `--nsteps` |
| Metropolis kT | 0.6 kcal/mol | `--kT` |
| Max rotation angle | 10 degrees | `--max_angle` |
| Random seed | 42 | `--seed` |

### Usage

```bash
# Single PDB
python scripts/run_backrub.py --pdb input.pdb --outdir ensembles/backrub --nconfs 5 --mode pyrosetta

# Job array mode (Wynton)
python scripts/run_backrub.py --pdb_list pdb_list.txt --task_id $SGE_TASK_ID \
    --outdir ensembles/backrub --nconfs 5 --mode pyrosetta
```

### CLI mode

An alternative `--mode cli` uses Rosetta's compiled `backrub` binary directly instead of PyRosetta. This follows the full Smith & Kortemme (2010) protocol with 75% backbone / 25% sidechain moves, Dunbrack rotamers, and 10% uniform chi sampling.

### Output structure

```
<outdir>/<stem>/
├── <stem>_backrub_000.pdb
├── <stem>_backrub_001.pdb
├── ...
├── trajectory_000.csv
├── trajectory_001.csv
└── ...
```

Each trajectory CSV contains columns: `step`, `score_current`, `score_lowest`, `accepted`.

---

## MC-SCE Protocol

**Script:** `scripts/run_mcsce.py`
**Submit script:** `scripts/submit_mcsce_wynton.sh`

### Protocol steps

1. **Preprocessing** (optional, `--preprocess`) — same idealize → repack → minimize pipeline as Backrub. Required for structures that have not been previously energy-minimized (e.g., raw AI-generated structures from ESMFold).
2. **Atom name normalization** — automatically converts Rosetta atom/residue names to AMBER conventions before passing to MCSCE:
   - `H1` → `H` (N-terminal hydrogen, non-PRO)
   - `H1` removed for N-terminal PRO (proline has no amide H)
   - `HIS_D`/`HSD` → `HID`, `HSE` → `HIE`, `HSP` → `HIP`
   - `MSE` → `MET` (with `SE` → `SD` atom rename)
   - `CYX`/`CYD` → `CYS`
3. **MC-SCE sampling** — strips side chains from backbone, then rebuilds them using Monte Carlo sampling with AMBER ff14SB energy (Lennard-Jones, clash, Coulomb terms). Produces N conformers with varied sidechain packing.
4. **Cartesian minimization** (optional, on by default) — Rosetta cartesian-space energy minimization (`beta_nov16_cart`) on each generated conformer with all DOFs (backbone + sidechain + jumps) enabled.

### Default parameters

| Parameter | Default | Flag |
|-----------|---------|------|
| Conformers per structure | 5 | `--nconfs` |
| Sampling temperature | 300 K | `--temperature` |
| Random seed | 42 | `--seed` |
| Preprocess input | off | `--preprocess` |
| Post-minimization | on | `--no_minimize` to skip |

### Usage

```bash
# Single PDB (with preprocessing for raw AI structures)
python scripts/run_mcsce.py --pdb input.pdb --outdir ensembles/mcsce --nconfs 100 --preprocess

# Job array mode (Wynton)
python scripts/run_mcsce.py --pdb_list pdb_list.txt --task_id $SGE_TASK_ID \
    --outdir ensembles/mcsce --nconfs 100 --preprocess --failed_log logs/mcsce_failed.txt
```

### Error handling

Structures that fail MC-SCE (non-standard residues, array indexing errors, force field incompatibilities) are caught and logged to the `--failed_log` file. The job continues rather than crashing.

### Output structure

```
<outdir>/<stem>/
├── 0.pdb
├── 1.pdb
├── ...
└── <n_conformers-1>.pdb
```

---

## HPC Submission

### Wynton (SGE)

```bash
# Backrub
qsub scripts/submit_backrub_wynton.sh

# MC-SCE
qsub scripts/submit_mcsce_wynton.sh
```

Both submit scripts use `#$ -t 1-N` for array jobs where N = number of PDBs in the list file. Each task processes one PDB via `$SGE_TASK_ID`.

### Key resource settings

| Resource | Backrub | MC-SCE |
|----------|---------|--------|
| Wall time | 4 hours | 15 hours |
| Memory | 8 GB | 48 GB |
| Scratch | — | 10 GB |
| Concurrent tasks | 200 | 200 |

---

## Dependencies

- **PyRosetta 4** (2024.39+) — Rosetta energy minimization, backrub sampling, structure preprocessing
- **MCSCE** ([THGLab/MCSCE](https://github.com/THGLab/MCSCE)) — side-chain ensemble generation with AMBER ff14SB
- Both are available in the `mcsce` conda environment on Wynton

---

## References

- Smith CA, Kortemme T. Backrub-like backbone simulation recapitulates natural protein conformational variability and improves mutant side-chain prediction. *J Mol Biol*. 2008;380(4):742-756.
- Teixeira JMC et al. MCSCE: Monte Carlo Side-Chain Entropy for protein structure prediction. *GitHub*: THGLab/MCSCE.
