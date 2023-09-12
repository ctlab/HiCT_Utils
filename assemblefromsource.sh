#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="${SCRIPT_DIR}/../HiCT_Library/"
DATA_DIR="${HICT_DIR}/../HiCT_Server/data/"
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
echo "Setting PYTHONPATH=$PYTHONPATH"
python -m hict_utils assemble $@ -c gzip -l 3 -a "${DATA_DIR}/kisumu_scaffolds_final.agp" "${DATA_DIR}/kisumu.fasta"