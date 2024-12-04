## Try running

- // 24.12.02: try running
```bash
# --
MAIN_DIR=/apdcephfs_qy3/share_733425/zhisonzhang/zh/2410critic/
_CACHE=${MAIN_DIR}/_cache/
export PYTHONPATH=${MAIN_DIR}/zopenrlhf
export TRANSFORMERS_CACHE=$_CACHE
export HF_DATASETS_CACHE=$_CACHE
conda activate openrlhf
# --
conda create -n openrlhf python=3.10
pip install torch==2.5.1 datasets peft --extra-index-url https://download.pytorch.org/whl/cu118
pip install accelerate bitsandbytes datasets deepspeed==0.15.0 einops flash-attn==2.7.0.post2 isort jsonlines loralib optimum packaging peft ray[default]==2.12.0 tensorboard torch torchmetrics tqdm transformers==4.46.3 transformers_stream_generator wandb wheel pandas
# --
yum install -y centos-release-scl
yum install -y devtoolset-9-gcc devtoolset-9-gcc-c++
source /opt/rh/devtoolset-9/enable
export CXX=/opt/rh/devtoolset-9/root/usr/bin/g++
export CC=/opt/rh/devtoolset-9/root/usr/bin/gcc
# --
# meta-llama/Meta-Llama-3.1-8B
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain Qwen/Qwen2.5-1.5B-Instruct \
   --save_path ./checkpoint/ \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing
EOF
#deepspeed --module $training_commands
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29512 python3 -mpdb -m $training_commands --local_rank=0
```
