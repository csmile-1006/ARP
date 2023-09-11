export experiment_name='MRDT-rtg-check'

ONLINE=True

## DT ONLY
USE_VL=${7}
VL_TYPE=${8}
VL_CHECKPOINT=${9}
USE_TASK_REWARD=False
USE_NORMALIZE=True

# --------------ENVIRONMENT------------------
GAME=${3}
DISTRIBUTION_MODE="hard"
START_LEVEL=0
NUM_LEVELS=500
DATA_ENV_TYPE=${4}
ENV_TYPE=${5}
USE_TRAIN_LEVELS=${6}

# Dataset
DATA_PATH=${2}
NUM_DEMONSTRATIONS=500
ENABLE_FILTER=True
NUM_FRAMES=8
WINDOW_SIZE=4

# --------------MODEL------------------
MODEL_TYPE="vit_base"
TRANSFER_TYPE="m3ae_vit_b16"
USE_TEXT=False
USE_ADAPTER=True
LOAD_CHECKPOINT="${1}"

# --------------EVALUATION------------------
NUM_TEST_EPISODES=100
EPISODE_LENGTH=500
RECORD_EVERY=10

# --------------COMMENT------------------
COMMENT="${10}"
NOTE="$COMMENT"
echo "note: $NOTE"

python3 -m arp_dt.local_run_procgen \
    --game_name="$GAME" \
    --load_checkpoint="$LOAD_CHECKPOINT" \
    --data.path="$DATA_PATH" \
    --data.train_env_type="$DATA_ENV_TYPE" \
    --data.num_demonstrations="$NUM_DEMONSTRATIONS" \
    --data.num_frames="$NUM_FRAMES" \
    --data.enable_filter="$ENABLE_FILTER" \
    --data.window_size="$WINDOW_SIZE" \
    --data.use_bert_tokenizer=True \
    --data.use_task_reward="$USE_TASK_REWARD" \
    --data.use_normalize="$USE_NORMALIZE" \
    --data.use_vl="$USE_VL" \
    --data.vl_type="$VL_TYPE" \
    --env.image_key="$IMAGE_KEY" \
    --env.distribution_mode="$DISTRIBUTION_MODE" \
    --env.start_level="$START_LEVEL" \
    --env.num_levels="$NUM_LEVELS" \
    --env.eval_env_type="$ENV_TYPE" \
    --env.record_every="$RECORD_EVERY" \
    --env.use_train_levels="$USE_TRAIN_LEVELS" \
    --model.model_type="$MODEL_TYPE" \
    --model.transfer_type="$TRANSFER_TYPE" \
    --model.use_adapter="$USE_ADAPTER" \
    --model.use_text="$USE_TEXT" \
    --use_text="$USE_TEXT"\
    --window_size="$WINDOW_SIZE" \
    --logging.online="$ONLINE" \
    --logging.prefix='' \
    --logging.project="$experiment_name" \
    --logging.output_dir="./eval_experiment_output/$experiment_name/$GAME" \
    --logging.random_delay=0.0 \
    --logging.notes="$NOTE" \
    --env.episode_length="$EPISODE_LENGTH" \
    --episode_length="$EPISODE_LENGTH" \
    --num_test_episodes="$NUM_TEST_EPISODES" \
    --use_normalize="$USE_NORMALIZE" \
    --use_vl="$USE_VL" \
    --vl_type="$VL_TYPE" \
    --use_task_reward="$USE_TASK_REWARD" \
    --vl_checkpoint="$VL_CHECKPOINT"