#!/bin/bash

#PBS -N ramses_training
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -j oe
#PBS -l select=1:mem=32G:ncpus=8:ngpus=1

cd $PBS_O_WORKDIR

module load cuda/12.4.1

source /mnt/lustre/helios-home/morovkat/miniconda3/etc/profile.d/conda.sh
conda activate ramses-trl-GPU

cd /mnt/lustre/helios-home/morovkat/ramses-trl_additional_processing

python train_model.py \
    --src-train /mnt/lustre/helios-home/morovkat/hiero-transformer/training_data/source_egy2tnt_cleaned.txt \
    --tgt-train /mnt/lustre/helios-home/morovkat/hiero-transformer/training_data/target_egy2tnt_cleaned.txt \
    --src-val /mnt/lustre/helios-home/morovkat/hiero-transformer/test_and_validation_data/validation_source_egy2tnt_cleaned.txt \
    --tgt-val /mnt/lustre/helios-home/morovkat/hiero-transformer/test_and_validation_data/validation_target_egy2tnt_cleaned.txt \
    --output /mnt/lustre/helios-home/morovkat/ramses-trl_additional_processing/network_UNICODE/hiero_transformer.h5

