#!/bin/bash

set -e
set -o pipefail

MODEL_DIRECTORY="/data2/the_beamers/image_captioning/data/top_down/"
MODEL="${MODEL_DIRECTORY}model-best.pth"
MODEL_INFOS_PATH="${MODEL_DIRECTORY}infos_td-best.pkl"
# If this is empty, will use the test set for MSCOCO
# IMAGE_DIRECTORY="/data2/the_beamers/image_captioning/data/val2014"
IMAGE_DIRECTORY='cute'
NUM_IMAGES=-1
BATCH_SIZE=1
BEAM_SIZE=10
OUTPUT_DIRECTORY="experimentsQualitative${BEAM_SIZE}"
LANG_EVAL=0

export CUDA_VISIBLE_DEVICES=2

mkdir -p $OUTPUT_DIRECTORY

echo "Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size $BEAM_SIZE \
              --output_json_file_path "$OUTPUT_DIRECTORY/bs"

echo "Beam Search, number of candidates=${BEAM_SIZE}, hidden_state_noise=0.3"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size $BEAM_SIZE \
              --hidden_state_noise 0.3 \
              --output_json_file_path "$OUTPUT_DIRECTORY/bs_npad0.3"

echo "Random sampling, temperature=1.0"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 1 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=1"

echo "Random sampling, temperature=0.7"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 0.7 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=0.7"

echo "Random sampling, temperature=0.5"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 0.5 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=1.3"

echo "Random sampling, top_c=10, temperature=1.0"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size 1 \
              --number_of_samples $BEAM_SIZE \
              --temperature 1.0 \
              --top_c 10 \
              --output_json_file_path "$OUTPUT_DIRECTORY/rs_t=1.0.top_c=10"
exit 
echo "K_per_Cand Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size $BEAM_SIZE \
              --k_per_cand 3 \
              --output_json_file_path "$OUTPUT_DIRECTORY/kpc"
 
echo "Hamming Penalty Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size $BEAM_SIZE \
              --hamming_penalty 0.8 \
              --output_json_file_path "$OUTPUT_DIRECTORY/hpbs"
# 
echo "Clustered Beam Search, number of candidates=${BEAM_SIZE}"
python eval.py --model "$MODEL" \
              --batch_size $BATCH_SIZE \
              --infos_path "$MODEL_INFOS_PATH" \
              --image_folder "$IMAGE_DIRECTORY" \
              --dump_images 0 \
              --language_eval $LANG_EVAL \
              --num_images $NUM_IMAGES \
              --sample_max 0 \
              --beam_size $BEAM_SIZE \
              --num_clusters 5 \
              --cluster_embeddings_file /data1/embeddings/eng/glove.42B.300d.txt \
              --output_json_file_path "$OUTPUT_DIRECTORY/cbs"
