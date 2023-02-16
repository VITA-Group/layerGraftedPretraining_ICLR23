# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
  # general
  -e|--epochs) epochs="$2"; shift; shift ;;
  -p|--port) port="$2"; shift; shift ;;
  -w|--workers) workers="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM="$2"; shift; shift ;;
  --batch_size) batch_size="$2"; shift; shift ;;
  --test_interval) test_interval="$2"; shift; shift ;;

  --data) data=("$2"); shift; shift ;;
  --arch) arch=("$2"); shift; shift ;;
  --pretrain) pretrain=("$2"); shift; shift ;;
  --pretrain_name) pretrain_name=("$2"); shift; shift ;;
  --save_dir) save_dir=("$2"); shift; shift ;;

  # finetune
  --fine_tune) fine_tune=("$2"); shift; shift ;;
  --customSplit) customSplit=("$2"); shift; shift ;;
  --customSplitName) customSplitName=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;
  --logstic_reg) logstic_reg=("$2"); shift; shift ;;
  --add_batch_norm) add_batch_norm=("$2"); shift; shift ;;

  # norm beit
  --norm_beit) norm_beit=("$2"); shift; shift ;;
  --sep_path) sep_path=("$2"); shift; shift ;;

  --skip_tune) skip_tune=("$2"); shift; shift ;;
  # feature_from
  --feature_from) feature_from=("$2"); shift; shift ;;

  --mae_aug) mae_aug=("$2"); shift; shift ;;
  --CAM) CAM=("$2"); shift; shift ;;
  --CAM_block) CAM_block=("$2"); shift; shift ;;
  --seed) seed=("$2"); shift; shift ;;

  --MI) MI=("$2"); shift; shift ;;
  --cosine_sim_entropy) cosine_sim_entropy=("$2"); shift; shift ;;

  --speed_test) speed_test=("$2"); shift; shift ;;

  --resume) resume=("$2"); shift; shift ;;

  --evaluate) evaluate=("$2"); shift; shift ;;

  --appendix) appendix=("$2"); shift; shift ;;

  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-90}
port=${port:-4833}
workers=${workers:-3}
GPU_NUM=${GPU_NUM:-1}
batch_size=${batch_size:-1024}
test_interval=${test_interval:-1}

data=${data:-'placeholder'}
arch=${arch:-'vit_small'}
pretrain=${pretrain:-"None"}
pretrain_name=${pretrain_name:-"None"}
save_dir=${save_dir:-"checkpoints_tune"}


fine_tune=${fine_tune:-"False"}
customSplit=${customSplit:-""}
customSplitName=${customSplitName:-""}
dataset=${dataset:-"imagenet100"}
logstic_reg=${logstic_reg:-"imagenet100"}
add_batch_norm=${add_batch_norm:-"False"}

norm_beit=${norm_beit:-"False"}
sep_path=${sep_path:-"0"}

skip_tune=${skip_tune:-"False"}
feature_from=${feature_from:-"-1"}

mae_aug=${mae_aug:-"False"}
CAM=${CAM:-"False"}
evaluate=${evaluate:-"False"}
CAM_block=${CAM_block:-"0"}
seed=${seed:-"None"}

MI=${MI:-"False"}
cosine_sim_entropy=${cosine_sim_entropy:-"False"}

speed_test=${speed_test:-"False"}

resume=${resume:-"None"}

appendix=${appendix:-""}

exp="linearSweep_B${batch_size}E${epochs}_${dataset}"

if [[ ${customSplit} != "" ]]
then
  exp=${exp}_${customSplitName}
fi

if [[ ${fine_tune} == "True" ]]
then
  exp=Tune_${exp}
fi

if [[ ${sep_path} != "0" ]]
then
  exp="${exp}_sepPath${sep_path}"
fi

if [[ ${speed_test} == "True" ]]
then
  exp=${exp}_testSpeed
fi

if [[ ${logstic_reg} == "True" ]]
then
  exp=${exp}_logReg
fi

if [[ ${add_batch_norm} == "True" ]]
then
  exp=${exp}_addBnClser
fi

if [[ ${norm_beit} == "True" ]]
then
  exp=${exp}_normBeit
fi

if [[ ${mae_aug} == "True" ]]
then
  exp="${exp}_maeAug"
fi

if [[ ${CAM} == "True" ]]
then
  exp="${exp}_CAM-B${CAM_block}"
fi

if [[ ${MI} == "True" ]]
then
  exp="${exp}_MI"
fi

if [[ ${cosine_sim_entropy} == "True" ]]
then
  exp="${exp}_cosineSimEntropy"
fi

if [[ ${feature_from} != "-1" ]]
then
  exp="${exp}_FeatFrom${feature_from}"
fi

if [[ ${appendix} != "" ]]
then
  exp=${exp}_${appendix}
fi

if [[ ${pretrain} != "None" ]]
then
  exp=${exp}__${pretrain_name}
fi

if [[ ${evaluate} == "True" ]]
then
  exp="${exp}_evaluate"
fi

launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"

cmd="${launch_cmd} main_lincls_sweep.py ${exp} --save_dir ${save_dir} -a ${arch} --epochs ${epochs} \
--multiprocessing-distributed --world-size 1 --rank 0 --batch-size ${batch_size} \
--data ${data} --dataset ${dataset} --workers ${workers} \
 --test-interval ${test_interval}"

if [[ ${pretrain} != "None" ]]
then
  cmd="${cmd} --pretrained ${pretrain}"
fi

if [[ ${seed} != "None" ]]
then
  cmd="${cmd} --seed ${seed}"
fi

if [[ ${evaluate} == "True" ]]
then
  cmd="${cmd} --evaluate"
fi

if [[ ${resume} != "None" ]]
then
  if [[ ${resume} == "local" ]]
  then
    resume=${save_dir}/${exp}/checkpoint.pth.tar
  fi
  cmd="${cmd} --resume ${resume}"
fi

if [[ ${customSplit} != "" ]]
then
  cmd="${cmd} --customSplit ${customSplit}"
fi

if [[ ${fine_tune} == "True" ]]
then
  cmd="${cmd} --fine-tune"
fi

if [[ ${sep_path} != "0" ]]
then
  cmd="${cmd} --sep-path ${sep_path}"
fi

if [[ ${logstic_reg} == "True" ]]
then
  cmd="${cmd} --logstic_reg"
fi

if [[ ${add_batch_norm} == "True" ]]
then
  cmd="${cmd} --add_batch_norm"
fi

if [[ ${norm_beit} == "True" ]]
then
  cmd="${cmd} --norm_beit"
fi

if [[ ${mae_aug} == "True" ]]
then
  cmd="${cmd} --mae_aug"
fi

if [[ ${CAM} == "True" ]]
then
  cmd="${cmd} --CAM --CAM_block ${CAM_block}"
fi

if [[ ${MI} == "True" ]]
then
  cmd="${cmd} --MI"
fi

if [[ ${cosine_sim_entropy} == "True" ]]
then
  cmd="${cmd} --cosine_sim_entropy"
fi

if [[ ${speed_test} == "True" ]]
then
  cmd="${cmd} --speed_test"
fi

if [[ ${feature_from} != "-1" ]]
then
  cmd="${cmd} --feature_from ${feature_from}"
fi

if [[ ${skip_tune} != "True" ]]
then
  echo ${cmd}
  ${cmd}
fi
