#!/bin/bash

set -e
set -o pipefail

MODEL_DIRECTORY="/data2/the_beamers/image_captioning/data/top_down/"
MODEL="${MODEL_DIRECTORY}model-best.pth"
MODEL_INFOS_PATH="${MODEL_DIRECTORY}infos_td-best.pkl"
IMAGE_DIRECTORY="/data2/the_beamers/image_captioning/data/val2017"
NUM_IMAGES=10
OUTPUT_DIRECTORY="experiments/"
BEAM_SIZE=10

mkdir -p OUTPUT_DIRECTORY

echo "Beam Search, number of candidates=${BEAM_SIZE}, hidden_state_noise=0.3"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 1 \
              --beam_size $BEAM_SIZE \
              --hidden_state_noise 0.3 \
              --output_json_file_path "$OUTPUT_DIRECTORY/bs_npad0.3.json"

echo "Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 1 \
              --beam_size $BEAM_SIZE \
              --output_json_file_path "$OUTPUT_DIRECTORY/bs.json"

echo "Random sampling, temperature=1.0"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 1 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=1.json"

echo "Random sampling, temperature=0.7"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 0.7 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=0.7.json"

echo "Random sampling, temperature=0.5"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 0.5 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=1.3.json"


echo "Random sampling, top_c=10, temperature=0.5"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 0.5 \
              --top_c 10 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=0.5,top_c=10.json"

echo "K_per_Cand Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 1 \
              --beam_size $BEAM_SIZE \
              --k_per_cand 3 \
              --output_json_file_path "$OUTPUT_DIRECTORY/kpc.json"

echo "Clustered Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --num_images $NUM_IMAGES \
              --sample_max 1 \
              --beam_size $BEAM_SIZE \
              --num_clusters 5 \
              --cluster_embeddings_file /data1/embeddings/eng/glove.42B.300d.txt \
              --output_json_file_path "$OUTPUT_DIRECTORY/cbs.json"