#!/bin/bash
#SBATCH --job-name=train-bert-token-tricksters
#SBATCH -t 12:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:2                    # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=16           # number cores per task
#SBATCH --mail-type=END,FAIL         # send mail when job begins and ends
#SBATCH --mail-user=l.kaesberg@stud.uni-goettingen.de   
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load anaconda3
module load cuda
source activate dnlp2 # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

git branch

# Run the script:
python -u multitask_classifier.py --use_gpu --option finetune --epochs 20 --sst_train data/ids-sst-train-syn2.csv
