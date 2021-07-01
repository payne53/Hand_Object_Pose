CUDA_VISIBLE_DEVICES=$1 \
python main.py \
    --input_file ./datasets/fhad/ \
	  --test \
	  --batch_size 64 \
	  --model_def ContextNet \
	  --gpu \
	  --gpu_number 0 \
	  --pretrained_model $2
