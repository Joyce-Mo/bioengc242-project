# bioengc242-project
Class project on mapping multi-conformer protein ensemble with diffusion models. 

# Github repo organization..
/scripts contains a bunch of random auxiliary scripts for data preparation and corresponding shell scripts to run them locally or on an HPC. 

/output contains some of the umaps of embeddings collected for this project. 
/logs contains some of the output and error files from sun grid engine or slurm jobs. 
/vae contains the VAE model for checkpoint #2. 

The flow-matching diffusion model is actually in a different repo, as a forked protpardelle that's been adapted for this specific project. The link to the diffusion model is here: https://github.com/Joyce-Mo/protpardelle-1c 

The created conformational emsembles are on box.com for storage.
