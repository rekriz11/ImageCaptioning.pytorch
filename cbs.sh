#!/bin/bash

set -e
set -o pipefail

MODEL_DIRECTORY="/data2/the_beamers/image_captioning/data/top_down/"
MODEL="${MODEL_DIRECTORY}model-best.pth"
MODEL_INFOS_PATH="${MODEL_DIRECTORY}infos_td-best.pkl"
IMAGE_DIRECTORY="/data2/the_beamers/image_captioning/data/val2017"
NUM_IMAGES=10
OUTPUT_DIRECTORY="experiments/"
BATCH_SIZE=10

mkdir -p OUTPUT_DIRECTORY

echo "Clustered Beam Search, number of candidates=${BATCH_SIZE}"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 1 \
              --beam_size $BATCH_SIZE \
              --num_clusters 5 \
              --cluster_embeddings_file /data1/embeddings/eng/glove.42B.300d.txt \
              --output_json_file_path "$OUTPUT_DIRECTORY/cbs.json"
