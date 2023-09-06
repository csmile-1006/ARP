export experiment_name='ARP-DT-procgen'

ONLINE=True
SEED=${9}

# DT ONLY
USE_VL=${6}
VL_TYPE=${7}
VL_CHECKPOINT=${8}
USE_TASK_REWARD=False
USE_NORMALIZE=True
INST_TYPE=${13:-"none"}

# Evaluation
GAME=${2}
TRAIN_ENV_TYPE=${3}
EVAL_ENV_TYPE=${4}

# Dataset
DATA_PATH=${1}
DISTRIBUTION_MODE="hard"
AUGMENTATIONS=${5}
START_LEVEL=0
NUM_LEVELS=500
NUM_DEMONSTRATIONS=500
ENABLE_FILTER=True
NUM_FRAMES=8
WINDOW_SIZE=4

# --------------MODEL------------------
MODEL_TYPE="vit_base"
TRANSFER_TYPE="m3ae_vit_b16"
USE_ADAPTER=True
LAMBDA_RETURN_PRED=${11}

# --------------TRAINING------------------
BATCH_SIZE=128
EPOCHS=50
TEST_EVERY_EPOCHS=20
LEARNING_RATE=5e-4
LR_SCHEDULE=cos
EVAL_WITH_GOAL=${12}

# --------------EVALUATION------------------
NUM_TEST_EPISODES=10
EPISODE_LENGTH=500
USE_TRAIN_LEVELS=False

# --------------COMMENT------------------
COMMENT=${10}
NOTE="$COMMENT"
echo "note: $NOTE"

python3 -m arp_dt.main_procgen \
    --is_tpu=False \
    --seed="$SEED" \
    --game_name="$GAME" \
    --data.path="$DATA_PATH" \
    --data.augmentations="$AUGMENTATIONS" \
    --data.num_demonstrations="$NUM_DEMONSTRATIONS" \
    --data.num_frames="$NUM_FRAMES" \
    --data.enable_filter="$ENABLE_FILTER" \
    --data.window_size="$WINDOW_SIZE" \
    --data.use_bert_tokenizer=True \
    --data.train_env_type="$TRAIN_ENV_TYPE" \
    --data.use_vl="$USE_VL" \
    --data.vl_type="$VL_TYPE" \
    --data.use_task_reward="$USE_TASK_REWARD" \
    --data.use_normalize="$USE_NORMALIZE" \
    --data.inst_type="$INST_TYPE" \
    --env.distribution_mode="$DISTRIBUTION_MODE" \
    --env.start_level="$START_LEVEL" \
    --env.num_levels="$NUM_LEVELS" \
    --env.eval_env_type="$EVAL_ENV_TYPE" \
    --env.use_train_levels="$USE_TRAIN_LEVELS" \
    --env.record_every="$NUM_TEST_EPISODES" \
    --env.episode_length="$EPISODE_LENGTH" \
    --model.model_type="$MODEL_TYPE" \
    --model.transfer_type="$TRANSFER_TYPE" \
    --model.use_adapter="$USE_ADAPTER" \
    --model.lambda_return_pred="$LAMBDA_RETURN_PRED" \
    --window_size="$WINDOW_SIZE" \
    --val_every_epochs=$(($EPOCHS / 5)) \
    --test_every_epochs=$(($EPOCHS / 5)) \
    --num_test_episodes="$NUM_TEST_EPISODES" \
    --batch_size="$BATCH_SIZE" \
    --weight_decay=5e-5 \
    --lr="$LEARNING_RATE" \
    --auto_scale_lr=False \
    --lr_schedule="$LR_SCHEDULE" \
    --warmup_epochs=10 \
    --momentum=0.9 \
    --clip_gradient=10.0 \
    --epochs="$EPOCHS" \
    --dataloader_n_workers=4 \
    --dataloader_shuffle=True \
    --log_all_worker=False \
    --logging.online="$ONLINE" \
    --logging.prefix='' \
    --logging.project="$experiment_name" \
    --logging.output_dir="$HOME/experiment_output/$experiment_name/${GAME}_${ENV_TYPE}" \
    --logging.random_delay=0.0 \
    --logging.notes="$NOTE" \
    --use_vl="$USE_VL" \
    --vl_type="$VL_TYPE" \
    --vl_checkpoint="$VL_CHECKPOINT" \
