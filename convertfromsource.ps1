Set-Variable "HICT_DIR" "${PSScriptRoot}/../HiCT_Library/"
$env:PYTHONPATH += ";${HICT_DIR}"
echo "Setting HICT_DIR = ${HICT_DIR} and PYTHONPATH = ${env:PYTHONPATH}"
# python -m hict_utils convert "../HiCT_Server/data/zanu_male_4DN.mcool"
# python -m hict_utils convert "D:/hi-c/zanu_male_4DN.mcool"
# python -m hict_utils convert "${HICT_DIR}/../HiCT_Server/data/mat18_100k.cool"
python -m hict_utils convert "${HICT_DIR}/../HiCT_Server/data/g3_4DN.mcool"
