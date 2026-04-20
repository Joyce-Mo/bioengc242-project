# bioengc242-project
Class project on mapping multi-conformer protein ensemble with diffusion models. 

# Project details (from proposal)

Current protein structure predictors, such as AlphaFold3 (AF3), infer native structures of proteins with high accuracy.  However, these models display poor diversity, limiting the ability to model ensembles of dynamic proteins or fit experimental data.  Recent efforts have shown that smaller models can approach the performance of AF3-type models, but yield better diversity and more realistic dynamics. This project is focused on the design of new protein folding models specific to the objective of mapping to multi-conformer datasets. This dataset includes multi-conformer ensembles fit to density maps from X-ray diffraction and cryo-EM structures (~40 PDBs) and synthetic data generated from Monte Carlo sampling for side-chain rotamers (15k PDB files from CATH20 + supplement from Monte Carlo side chain entropy method by Asmit Bhowmick and Teresa Head-Gordon ). 

I will first build a larger dataset of synthetic conformers with MC-SCE, and then assess the distribution of the datasets at hand with foldseek. Then, I will build 2 models for multi-conformer mapping: (1) flow-matching diffusion, and (2) variational autoencoder (VAE) coupled to a score-matching diffusion model. 

Flow-matching diffusion architectures may perform better than off-the-shelf foundation models like Boltz2 for my specific objective, since flow-matching training involves a learned vector field for a “wider” set of conformations induced by the noising process, rather than a single, native state structure. Proteina is similarly a flow-based protein generative model; however, this is specific to protein backbones.  In my case, I am interested in sidechains and backbones, and will follow a similar superpositioning process as what was done for Protpardelle. For this project, I will adapt protpardelle for flow-matching loss and I will distill representations from "expert" models, like Boltz2, MPNN, and Frame2Seq. This involves a convolutional projection of the features from an expert model and the "student" Protpardelle model and centered kernel alignment loss between the features. The novelty of this model is the combination of a vision-transformer (Protpardelle is an off-the-shelf U-ViT) with flow-matching diffusion, and representation alignment of this new protpardelle with embeddings from expert models. 

Regarding the second model architecture, the VAE will be trained to encode spatial features (amino acid types, such as polarity, hydrophobicity, charge), kernel alignment distances between side chain residues and backbone, and SASA to create latent embeddings for improved protein structure generation. In my case, I am examining multi-conformer protein structures.

Both models will be trained on the AI-CATH (from Protpardelle-1c) and multi-conformer datasets. I will evaluate the models with shared results from the Fraser lab with guidance directed towards density maps. 

# Synthetic data generation
I will explore 2 methods for obtaining side-chain rotamers from the ai-cath dataset. 

1. MC-SCE from Head-Gordon's lab
2. Backrub from RosettaCommons/Kortemme lab. 

Both use monte carlo methods and are less computationally expensive as REMD. 

# Model architectures: 
1. I will implement a flow-based diffusion model based on Protpardelle's architecture
2. I will implement a VAE and score-based diffusion. 

# Github repo organization..
This is a work in progress!! Until the project submission. For now, scripts contains a bunch of random auxiliary scripts for data preparation and corresponding shell scripts to run them locally or on an HPC. 

/output contains some of the umaps of embeddings collected for this project. 
/logs contains some of the output and error files from sun grid engine or slurm jobs. 
/vae contains the VAE model for checkpoint #2. 

The flow-matching diffusion model is actually in a different repo, as a forked protpardelle that's been adapted for this specific project. 

The created conformational emsembles are on box.com for storage.

# HPC notes

## Expanse (SDSC)
- Account: `ucb368`, allocation is on **gpu-shared** partition only. The `shared` (CPU-only) partition will reject jobs with "Project not found".
- All batch jobs (even CPU-only work like featurization or tar archiving) must use `-p gpu-shared` with `--gpus=1`.
- `ucb368` has a low `MaxSubmitJobsPerAccount` limit. Keep array sizes small (e.g. `--array=1-16`). Even with a `%N` throttle, SLURM counts all pending array tasks against the submit limit.
- Long-running commands (tar, featurization) must be submitted as batch jobs, not run on the login node. SSH sessions to Expanse time out and kill foreground processes.