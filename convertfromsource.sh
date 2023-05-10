#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
echo "Setting PYTHONPATH=$PYTHONPATH"
# python -m hict_utils convert "../HiCT_Server/data/zanu_male_4DN.mcool"
# python -m hict_utils convert "D:/hi-c/zanu_male_4DN.mcool"
# python -m hict_utils convert "${HICT_DIR}/../HiCT_Server/data/mat18_100k.cool"
# python -m hict_utils convert "${HICT_DIR}/../HiCT_Server/data/g3_4DN.mcool"
python -m hict_utils convert "${HICT_DIR}/../HiCT_Server/data/zanu_male_4DN.mcool"
