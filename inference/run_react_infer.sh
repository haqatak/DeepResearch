#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1

##############hyperparams################
export OLLAMA_MODEL_NAME="huggingface.co/gabriellarson/Tongyi-DeepResearch-30B-A3B-GGUF"
export DATASET=example
export OUTPUT_PATH=./output
export ROLLOUT_COUNT=3 # eval avg@3
export TEMPERATURE=0.85
export PRESENCE_PENALTY=1.1
export MAX_WORKERS=30


## serper key for search&google scholar
## https://serper.dev/
# export SERPER_KEY_ID="" # No longer needed, using DuckDuckGo

## jina key for read page
## https://jina.ai/
# export JINA_API_KEYS="" # No longer needed, using Playwright

## summary model api for page summary in visit tool
## https://platform.openai.com/
export API_KEY=""
export API_BASE=""
export SUMMARY_MODEL_NAME=""

## dashscope key for file parser
## https://dashscope.aliyun.com/
export DASHSCOPE_API_KEY=""  # support：qwen-omni-turbo，qwen-plus-latest
export DASHSCOPE_API_BASE=""
export VIDEO_MODEL_NAME=""
export VIDEO_ANALYSIS_MODEL_NAME=""

# code sandbox ip for python interperter
# example for ENDPOINTS_STRING "http://22.16.67.220:8080,http://22.16.78.153:8080,http://22.17.10.216:8080,http://22.14.58.9:8080,http://22.16.14.3:8080,http://22.17.26.164:8080,http://22.16.245.207:8080"
# we use sandbox_fusion: https://github.com/bytedance/SandboxFusion
ENDPOINTS_STRING=""
export SANDBOX_FUSION_ENDPOINT="$ENDPOINTS_STRING"
export TORCH_COMPILE_CACHE_DIR="./cache"

# IDP service is used for file parsing. If set to false, rule-based parsing is used. You can add an IDP key and set USE_IDP=True to use a more powerful file parsing tool.
# https://help.aliyun.com/zh/document-mind/developer-reference/use-idp-llm-to-complete-document-summary
export USE_IDP=False
export IDP_KEY_ID=""
export IDP_KEY_SECRET=""

#####################################
### 1. start infer               ####
#####################################

echo "==== start infer... ===="
echo "Please ensure the ollama server is running."


cd "$( dirname -- "${BASH_SOURCE[0]}" )"

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model "$OLLAMA_MODEL_NAME" --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT
