#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
echo "Setting PYTHONPATH=$PYTHONPATH"
DATA_DIR="${HICT_DIR}/../HiCT_Server/data/"

docker stop higlass-container;
docker rm higlass-container;

docker pull higlass/higlass-docker:v0.6.1 # higher versions are experimental and may or may not work


docker run --detach \
           --publish 8989:80 \
           --volume "$DATA_DIR:/data" \
           --volume ~/tmp:/tmp \
           --name higlass-container \
         higlass/higlass-docker:v0.6.1