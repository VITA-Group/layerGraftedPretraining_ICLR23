# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
  # general
  -e|--epochs) epochs="$2"; shift; shift ;;
  -g|--GPU_NUM) GPU_NUM="$2"; shift; shift ;;
  -s|--split) split="$2"; shift; shift ;;
  -p|--port) port="$2"; shift; shift ;;
  --MASTER) MASTER="$2"; shift; shift ;;
  --NODE_NUM) NODE_NUM="$2"; shift; shift ;;
  --RANK) RANK="$2"; shift; shift ;;
  --msr) msr="$2"; shift; shift ;;
  --world_size) world_size="$2"; shift; shift ;;
  --skip_tune) skip_tune="$2"; shift; shift ;;
  --skip_finetune) skip_finetune="$2"; shift; shift ;;
  --skip_pretrain) skip_pretrain="$2"; shift; shift ;;

  --workers) workers="$2"; shift; shift ;;
  --warm_up_epochs) warm_up_epochs="$2"; shift; shift ;;
  --data) data=("$2"); shift; shift ;;
  --arch) arch=("$2"); shift; shift ;;
  --batch_size) batch_size=("$2"); shift; shift ;;
  --lr) lr=("$2"); shift; shift ;;
  --tune_lr) tune_lr=("$2"); shift; shift ;;
  --tune_batch_size) tune_batch_size=("$2"); shift; shift ;;
  --few_shot_tune_lr) few_shot_tune_lr=("$2"); shift; shift ;;
  --few_shot_tune_batch_size) few_shot_tune_batch_size=("$2"); shift; shift ;;
  --mae_tune_lr) mae_tune_lr=("$2"); shift; shift ;;
  --temp) temp=("$2"); shift; shift ;;
  --save_dir) save_dir=("$2"); shift; shift ;;
  --ckpt_pretrain) ckpt_pretrain=("$2"); shift; shift ;;
  --ckpt_tune) ckpt_tune=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;

  # test option
  --resume) resume=("$2"); shift; shift ;;
  --resume_appendix) resume_appendix=("$2"); shift; shift ;;
  --appendix) appendix=("$2"); shift; shift ;;
  --evaluate_pretrain) evaluate_pretrain=("$2"); shift; shift ;;
  --evaluate_pretrain_representation) evaluate_pretrain_representation=("$2"); shift; shift ;;
  --evaluate_moe_gate_selection) evaluate_moe_gate_selection=("$2"); shift; shift ;;
  --evaluate_moe_gate_selection_fix_trans) evaluate_moe_gate_selection_fix_trans=("$2"); shift; shift ;;

  # cmae options
  --cmae) cmae=("$2"); shift; shift ;;

  # VIC_reg
  --VIC_version) VIC_version=("$2"); shift; shift ;;

  # the local crops number
  --local_crops_number) local_crops_number=("$2"); shift; shift ;;

  --eval_gradient) eval_gradient=("$2"); shift; shift ;;

  # grafting
  --graft_pretrained) graft_pretrained=("$2"); shift; shift ;;
  --graft_pretrained_name) graft_pretrained_name=("$2"); shift; shift ;;
  --scratch_end) scratch_end=("$2"); shift; shift ;;
  --graft_CL_end) graft_CL_end=("$2"); shift; shift ;;
  --end_pretrained) end_pretrained=("$2"); shift; shift ;;
  --fixTo) fixTo=("$2"); shift; shift ;;
  --l1_dist_w) l1_dist_w=("$2"); shift; shift ;;
  --l1_dist_to_block) l1_dist_to_block=("$2"); shift; shift ;;

  # conditioned_predictor
  --conditioned_predictor) conditioned_predictor=("$2"); shift; shift ;;
  --conditioned_predictor_temp) conditioned_predictor_temp=("$2"); shift; shift ;;

  # sim_rank
  --sim_rank_alpha) sim_rank_alpha=("$2"); shift; shift ;;

  # layer_wise_decay_ratio
  --layer_wise_decay_ratio) layer_wise_decay_ratio=("$2"); shift; shift ;;
  --lr_layer_wise) lr_layer_wise=("$2"); shift; shift ;;

  # mae aug for moco
  --mae_aug) mae_aug=("$2"); shift; shift ;;

  # evaluate methods
  --evaluate_VIC) evaluate_VIC=("$2"); shift; shift ;;
  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-200}
GPU_NUM=${GPU_NUM:-1}
split=${split:-full}
port=${port:-4833}
NODE_NUM=${NODE_NUM:-1}
msr=${msr:-"False"}
world_size=${world_size:-"1"}
skip_tune=${skip_tune:-"False"}
skip_finetune=${skip_finetune:-"False"}
skip_pretrain=${skip_pretrain:-"False"}

workers=${workers:-6}
warm_up_epochs=${warm_up_epochs:-40}
data=${data:-'placeholder'}
arch=${arch:-'vit_small'}
batch_size=${batch_size:-1024}
lr=${lr:-"1.5e-4"}
temp=${temp:-0.2}
save_dir=${save_dir:-"checkpoints"}
ckpt_pretrain=${ckpt_pretrain:-"checkpoints"}
ckpt_tune=${ckpt_tune:-"checkpoints_tune"}
dataset=${dataset:-"imagenet100"}

resume=${resume:-""}
resume_appendix=${resume_appendix:-""}
appendix=${appendix:-""}
evaluate_pretrain=${evaluate_pretrain:-""}
evaluate_pretrain_representation=${evaluate_pretrain_representation:-""}

cmae=${cmae:-"False"}

VIC_version=${VIC_version:-"False"}

local_crops_number=${local_crops_number:-"0"}

eval_gradient=${eval_gradient:-"False"}

graft_pretrained=${graft_pretrained:-""}
graft_pretrained_name=${graft_pretrained_name:-""}
scratch_end=${scratch_end:-"False"}
graft_CL_end=${graft_CL_end:-"False"}
end_pretrained=${end_pretrained:-""}
conditioned_predictor=${conditioned_predictor:-"False"}
conditioned_predictor_temp=${conditioned_predictor_temp:-"False"}
layer_wise_decay_ratio=${layer_wise_decay_ratio:-"-1"}
lr_layer_wise=${lr_layer_wise:-""}
mae_aug=${mae_aug:-"False"}
evaluate_VIC=${evaluate_VIC:-"False"}
fixTo=${fixTo:-"-1"}
l1_dist_w=${l1_dist_w:-"-1"}
l1_dist_to_block=${l1_dist_to_block:-"4"}

sim_rank_alpha=${sim_rank_alpha:-"-1"}

if [[ ${dataset} == "imagenet" ]]
then
  tune_lr=${tune_lr:-"3.0"}
  tune_batch_size=${tune_batch_size:-"4096"}
  few_shot_tune_batch_size=${few_shot_tune_batch_size:-"256"}
  few_shot_tune_epochs=${few_shot_tune_epochs:-"800"}
else
  tune_lr=${tune_lr:-"1.0"}
  tune_batch_size=${tune_batch_size:-"1024"}
  few_shot_tune_batch_size=${few_shot_tune_batch_size:-"256"}
  few_shot_tune_epochs=${few_shot_tune_epochs:-"800"}
fi

mae_tune_lr=${mae_tune_lr:-"5e-4"}

exp_name="${arch}_${dataset}_lr${lr}B${batch_size}E${epochs}"

if [[ ${local_crops_number} != "0" ]]
then
  exp_name="${exp_name}_localCropsN${local_crops_number}"
fi

if [[ ${eval_gradient} == "True" ]]
then
  exp_name="${exp_name}_evalGrad"
fi

if [[ ${graft_pretrained} != "" ]]
then
  exp_name="${exp_name}_graftF${graft_pretrained_name}"
fi

if [[ ${scratch_end} == "True" ]]
then
  exp_name="${exp_name}_scratchEnd"
fi

if [[ ${graft_CL_end} == "True" ]]
then
  exp_name="${exp_name}_CLend"
fi

if [[ ${conditioned_predictor} == "True" ]]
then
  exp_name="${exp_name}_condPred"
  if [[ ${conditioned_predictor_temp} == "True" ]]
  then
    exp_name="${exp_name}Temp"
  fi
fi

if [[ ${layer_wise_decay_ratio} != "-1" ]]
then
  exp_name="${exp_name}_layerWiseD${layer_wise_decay_ratio}"
fi

if [[ ${lr_layer_wise} != "" ]]
then
  exp_name="${exp_name}_lrLayerW${lr_layer_wise}"
fi

if [[ ${mae_aug} == "True" ]]
then
  exp_name="${exp_name}_maeAug"
fi

if [[ ${evaluate_VIC} == "True" ]]
then
  exp_name="${exp_name}_evalVIC"
fi

if [[ ${fixTo} != "-1" ]]
then
  exp_name="${exp_name}_fixTo${fixTo}"
fi

if [[ ${l1_dist_w} != "-1" ]]
then
  exp_name="${exp_name}_l1W${l1_dist_w}To${l1_dist_to_block}"
fi

if [[ ${sim_rank_alpha} != "-1" ]]
then
  exp_name="${exp_name}_simRankW${sim_rank_alpha}"
fi

if [[ ${resume_appendix} != "" ]] && [[ ${resume} != "" ]]
then
  exp_name="${exp_name}__${resume_appendix}"
fi

if [[ ${cmae} == "True" ]]
then
  exp_name="${exp_name}_cmae"
fi

if [[ ${VIC_version} == "True" ]]
then
  exp_name="${exp_name}_VIC"
fi

if [[ ${appendix} != "" ]]
then
  exp_name="${exp_name}_${appendix}"
fi

if [[ ${NODE_NUM} != "1" ]]
then
  launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} \
  --nnodes=${NODE_NUM} --node_rank=${RANK} --master_addr=${MASTER}"
elif [[ ${msr} == "True" ]]
then
  launch_cmd="python"
else
  launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"
fi

cmd="${launch_cmd} main_pretrain.py ${exp_name} \
  -a ${arch} -b ${batch_size} \
  --optimizer=adamw --lr=${lr} --weight-decay=.1 \
  --epochs=${epochs} --warmup-epochs=${warm_up_epochs} \
  --stop-grad-conv1 --moco-m-cos --moco-t=${temp} \
  --dist-url tcp://localhost:${port} \
  --multiprocessing-distributed --world-size ${world_size} --rank 0 \
  --dataset ${dataset} --workers ${workers} \
  --save_dir ${save_dir}/${ckpt_pretrain} --data ${data}"


############## linear evaluation cmd ##############
fine_tune_cmd="bash cmds/shell_scripts/tunePretrained_sweep.sh \
--pretrain_name ${exp_name} --dataset ${dataset} --batch_size ${tune_batch_size} \
--pretrain ${save_dir}/${ckpt_pretrain}/${exp_name}/checkpoint_final.pth.tar \
--save_dir ${save_dir}/${ckpt_tune} --arch ${arch} -g ${GPU_NUM} -p ${port} --data ${data}"

if [[ ${cmae} == "True" ]]
then
  cmd="${cmd} --cmae"
fi

if [[ ${VIC_version} == "True" ]]
then
  cmd="${cmd} --VIC_version --moco-dim 8192 --moco-mlp-dim 8192"
fi

if [[ ${layer_wise_decay_ratio} != "-1" ]]
then
  cmd="${cmd} --layer_wise_decay_ratio ${layer_wise_decay_ratio}"
fi

if [[ ${lr_layer_wise} != "" ]]
then
  cmd="${cmd} --lr_layer_wise ${lr_layer_wise}"
fi

if [[ ${mae_aug} == "True" ]]
then
  cmd="${cmd} --mae_aug"
fi

if [[ ${evaluate_VIC} == "True" ]]
then
  cmd="${cmd} --evaluate_VIC"
fi

if [[ ${local_crops_number} != "0" ]]
then
  cmd="${cmd} --local_crops_number ${local_crops_number}"
fi

if [[ ${eval_gradient} == "True" ]]
then
  cmd="${cmd} --eval_gradient"
fi

if [[ ${graft_pretrained} != "" ]]
then
  cmd="${cmd} --graft_pretrained ${graft_pretrained}"
fi

if [[ ${scratch_end} == "True" ]]
then
  cmd="${cmd} --scratch_end"
fi

if [[ ${graft_CL_end} == "True" ]]
then
  cmd="${cmd} --graft_CL_end --end_pretrained ${end_pretrained}"
fi

if [[ ${conditioned_predictor} == "True" ]]
then
  cmd="${cmd} --conditioned_predictor"
  if [[ ${conditioned_predictor_temp} == "True" ]]
  then
    cmd="${cmd} --conditioned_predictor_temp"
  fi
fi

if [[ ${fixTo} != "-1" ]]
then
  cmd="${cmd} --fixTo ${fixTo}"
fi

if [[ ${l1_dist_w} != "-1" ]]
then
  cmd="${cmd} --l1_dist_w ${l1_dist_w} --l1_dist_to_block ${l1_dist_to_block}"
fi

if [[ ${sim_rank_alpha} != "-1" ]]
then
  cmd="${cmd} --sim_rank_alpha ${sim_rank_alpha} --sim_rank"
fi

resume_tune="False"
if [[ ${resume} != "" ]]
then
  if [[ ${resume} == "local" ]]
  then
    resume=${save_dir}/${ckpt_pretrain}/${exp_name}/checkpoint_last.pth.tar
    resume_tune="True"
  fi
  cmd="${cmd} --resume ${resume}"
fi

if [[ ${evaluate_pretrain} != "" ]]
then
  cmd="${cmd} --evaluate_pretrain"
  if [[ ${evaluate_pretrain_representation} != "" ]]
  then
    cmd="${cmd} --evaluate_pretrain_representation"
  fi
fi

if [[ ${evaluate_moe_gate_selection} == "True" ]]
then
  cmd="${cmd} --evaluate_moe_gate_selection"
  if [[ ${evaluate_moe_gate_selection_fix_trans} == "True" ]]
  then
    cmd="${cmd} --evaluate_moe_gate_selection_fix_trans"
  fi
fi

################### mae tune cmd ###################
mae_tune_cmd="bash cmds/shell_scripts/tunePretrained_mae.sh \
--pretrain_name ${exp_name} --dataset ${dataset} --batch_size 1024 --lr ${mae_tune_lr} \
--pretrain ${save_dir}/${ckpt_pretrain}/${exp_name}/checkpoint_final.pth.tar \
--save_dir ${save_dir}/${ckpt_tune} --arch ${arch} -g ${GPU_NUM} -p ${port} --data ${data}"


if [[ ${resume_tune} == "True" ]]
then
  fine_tune_cmd="${fine_tune_cmd} --resume local"
  mae_tune_cmd="${mae_tune_cmd} --resume local"
fi

###################################################
if [[ ${skip_pretrain} != "True" ]]
then
echo ${cmd}
${cmd}
fi

if [[ ${skip_tune} != "True" ]] && [[ ${msr} != "True" ]]
then
  echo ${fine_tune_cmd}
  ${fine_tune_cmd}

  if [[ ${skip_finetune} != "True" ]]
  then
    echo ${mae_tune_cmd}
    ${mae_tune_cmd}
  fi
fi