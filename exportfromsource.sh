#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
echo "Setting PYTHONPATH=$PYTHONPATH"
python -m hict_utils export "${HICT_DIR}/../HiCT_Server/data/mat18_100k.mcool.hict.hdf5"
# python -m hict_utils export --agp "${HICT_DIR}/../HiCT_Server/data/g3_scaffolds_final.agp" "${HICT_DIR}/../HiCT_Server/data/g3_4DN.mcool.hict.hdf5"
# python -m hict_utils export "${HICT_DIR}/../HiCT_Server/data/zanu_male_4DN.mcool.hict.hdf5"
