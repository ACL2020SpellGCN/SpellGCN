#!/usr/bin/env bash
if [ $# -lt 6 ];then
  echo 'Usage sh run_spellgcn.sh <DATA_DIR> <JOB_NAME> <BERT_PATH> <SIGHAN13_PATH> <SIGHAN14_PATH> <SIGHAN15_PATH>'
  exit 0
fi

task_name=CSC
timestamp=`date "+%Y-%m-%d-%H-%M-%S"`
lr=5e-5
batch_size=32
num_epochs=10
max_seq_length=180
do_lower_case=true
graph_dir="../data/gcn_graph.ty_xj/"
 
mkdir -p log/

# TRAIN
for i in $(seq 0 0)
do

output_dir=log/${2}_sighan13_${task_name}_$i 
log_dir=log/${2}_sighan13_${task_name}_$i 

if [ ! -d "${output_dir}/src" ]; then
  mkdir -p ${output_dir}/src
  cp $0 ${output_dir}/src
  cp ../*py ${output_dir}/src
fi

#sleep $i
echo "Start running ${task_name} task-${i} log to ${output_dir}.log"
CUDA_VISIBLE_DEVICES=$i python ../run_spellgcn.py \
  --job_name=$2 \
  --task_name=${task_name} \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --data_dir=$1 \
  --vocab_file=$3/vocab.txt \
  --bert_config_file=$3/bert_config.json \
  --max_seq_length=${max_seq_length} \
  --max_predictions_per_seq=${max_seq_length} \
  --train_batch_size=${batch_size} \
  --learning_rate=${lr} \
  --num_train_epochs=${num_epochs} \
  --keep_checkpoint_max=10 \
  --random_seed=${i}000 \
  --init_checkpoint=$3/bert_model.ckpt \
  --graph_dir=${graph_dir} \
  --output_dir=${output_dir} > ${log_dir}.log 2>&1 &
done
wait

# PREDICT & TEST
for i in $(seq 0 0)
do

output_dir=log/${2}_sighan13_${task_name}_$i 
log_dir=log/${2}_sighan13_${task_name}_$i 

CUDA_VISIBLE_DEVICES=$i python ../run_spellgcn.py \
  --job_name=$2 \
  --task_name=${task_name} \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --data_dir=$4 \
  --vocab_file=$3/vocab.txt \
  --bert_config_file=$3/bert_config.json \
  --max_seq_length=${max_seq_length} \
  --max_predictions_per_seq=${max_seq_length} \
  --train_batch_size=${batch_size} \
  --learning_rate=${lr} \
  --num_train_epochs=${num_epochs} \
  --keep_checkpoint_max=10 \
  --random_seed=${i}000 \
  --init_checkpoint=${output_dir} \
  --graph_dir=${graph_dir} \
  --output_dir=${output_dir} >> ${log_dir}.log 2>&1 &
done
wait

# PREDICT & TEST
#for i in $(seq 0 0)
#do

#output_dir=log/${2}_sighan13_${task_name}_$i 
#log_dir=log/${2}_sighan14_${task_name}_$i 

#CUDA_VISIBLE_DEVICES=$i python ../run_spellgcn.py \
#  --job_name=$2 \
#  --task_name=${task_name} \
#  --do_train=False \
#  --do_eval=False \
#  --do_predict=True \
#  --data_dir=$5 \
#  --vocab_file=$3/vocab.txt \
#  --bert_config_file=$3/bert_config.json \
#  --max_seq_length=${max_seq_length} \
#  --max_predictions_per_seq=${max_seq_length} \
#  --train_batch_size=${batch_size} \
#  --learning_rate=${lr} \
#  --num_train_epochs=${num_epochs} \
#  --keep_checkpoint_max=10 \
#  --random_seed=${i}000 \
#  --init_checkpoint=${output_dir} \
#  --graph_dir=${graph_dir} \
#  --output_dir=${output_dir} >> ${log_dir}.log 2>&1 &
#done
#wait

## PREDICT & TEST
#for i in $(seq 0 0)
#do

#output_dir=log/${2}_sighan13_${task_name}_$i 
#log_dir=log/${2}_sighan15_${task_name}_$i 

#CUDA_VISIBLE_DEVICES=$i python ../run_spellgcn.py \
#  --job_name=$2 \
#  --task_name=${task_name} \
#  --do_train=False \
#  --do_eval=False \
#  --do_predict=True \
#  --data_dir=$6 \
#  --vocab_file=$3/vocab.txt \
#  --bert_config_file=$3/bert_config.json \
#  --max_seq_length=${max_seq_length} \
#  --max_predictions_per_seq=${max_seq_length} \
#  --train_batch_size=${batch_size} \
#  --learning_rate=${lr} \
#  --num_train_epochs=${num_epochs} \
#  --keep_checkpoint_max=10 \
#  --random_seed=${i}000 \
#  --init_checkpoint=${output_dir} \
#  --graph_dir=${graph_dir} \
#  --output_dir=${output_dir} >> ${log_dir}.log 2>&1 &
#done
