RUN_SCRIPT=../run_rcv1_label_descriptions.py
OUTPUT_DIR=./model_outputs/semsup_single

# Default command
BOOL_ARGS="--do_train --do_eval --do_predict --use_label_descriptions --overwrite_output_dir"
DEFAULT_COMMAND="python $RUN_SCRIPT --cache_dir ../../data_cache --max_seq_length 256 --per_device_train_batch_size 24 --per_device_eval_batch_size 12 --learning_rate 2e-5 --lr_scheduler_type constant --num_train_epochs 2 --save_steps -1 --evaluation_strategy steps --eval_steps 10000 --all_labels_file ../rcv1/rcv1_all.json --hierarchy_file ../rcv1.topics.hier.orig --output_dir $OUTPUT_DIR $BOOL_ARGS"

# Files being used
TRAIN_FILE=../rcv1/rcv1_train.json
VALIDATION_FILE=../rcv1/rcv1_val.json
TEST_FILE=../rcv1/rcv1_test.json

TRAIN_DESCRIPTIONS=../../class_descrs/rcv1/s1_unseen_descriptions/train_descriptions_n_1.json
VAL_DESCRIPTIONS=../../class_descrs/rcv1/s1_unseen_descriptions/train_descriptions_n_1.json
TEST_DESCRIPTIONS=../../class_descrs/rcv1/s1_unseen_descriptions/val_descriptions.json

# Run the command
MODEL_NAME="distilbert-base-multilingual-cased"
$DEFAULT_COMMAND --model_name_or_path $MODEL_NAME --label_model_name_or_path $MODEL_NAME --train_file $TRAIN_FILE --validation_file $VALIDATION_FILE --test_file $TEST_FILE --label_descriptions_train_file $TRAIN_DESCRIPTIONS --label_descriptions_validation_file $VAL_DESCRIPTIONS --label_descriptions_test_file $TEST_DESCRIPTIONS