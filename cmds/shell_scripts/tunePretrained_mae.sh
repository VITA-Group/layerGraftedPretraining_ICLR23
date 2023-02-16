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
  --accum_iter) accum_iter="$2"; shift; shift ;;
  --world_size) world_size="$2"; shift; shift ;;
  --test_interval) test_interval="$2"; shift; shift ;;
  --drop_path) drop_path=("$2"); shift; shift ;;

  --data) data=("$2"); shift; shift ;;
  --dataset) dataset=("$2"); shift; shift ;;
  --arch) arch=("$2"); shift; shift ;;
  --lr) lr=("$2"); shift; shift ;;
  --layer_decay) layer_decay=("$2"); shift; shift ;;
  --wd) wd=("$2"); shift; shift ;;
  --pretrain) pretrain=("$2"); shift; shift ;;
  --pretrain_name) pretrain_name=("$2"); shift; shift ;;
  --save_dir) save_dir=("$2"); shift; shift ;;
  --eval_gradient) eval_gradient=("$2"); shift; shift ;;
  --eval_gradient_eval_aug) eval_gradient_eval_aug=("$2"); shift; shift ;;
  --fixTo) fixTo=("$2"); shift; shift ;;
  --fixBnStat) fixBnStat=("$2"); shift; shift ;;
  --add1BlockTo) add1BlockTo=("$2"); shift; shift ;;
  --add1BlockToNumLayers) add1BlockToNumLayers=("$2"); shift; shift ;;
  --reset_layers) reset_layers=("$2"); shift; shift ;;
  --add_batch_norm) add_batch_norm=("$2"); shift; shift ;;

  --customSplit) customSplit=("$2"); shift; shift ;;
  --customSplitName) customSplitName=("$2"); shift; shift ;;
  --tuneFromFirstFC) tuneFromFirstFC=("$2"); shift; shift ;;
  --eval_avg_attn_dist) eval_avg_attn_dist=("$2"); shift; shift ;;

  --fc_scale) fc_scale=("$2"); shift; shift ;;

  --no_aug) no_aug=("$2"); shift; shift ;;
  --cutmix) cutmix=("$2"); shift; shift ;;
  --mixup) mixup=("$2"); shift; shift ;;
  --reprob) reprob=("$2"); shift; shift ;;
  --aa) aa=("$2"); shift; shift ;;

  --norm_beit) norm_beit=("$2"); shift; shift ;;

  # avg attn dist reg
  --avg_attn_dist_reg) avg_attn_dist_reg=("$2"); shift; shift ;;
  --avg_attn_dist_reg_target) avg_attn_dist_reg_target=("$2"); shift; shift ;;
  --avg_attn_dist_reg_target_name) avg_attn_dist_reg_target_name=("$2"); shift; shift ;;

  --resume) resume=("$2"); shift; shift ;;
  --msr) msr=("$2"); shift; shift ;;

  *) echo "${1} is not found"; exit 125;
esac
done

epochs=${epochs:-100}
port=${port:-4833}
workers=${workers:-4}
GPU_NUM=${GPU_NUM:-1}
batch_size=${batch_size:-1024}
accum_iter=${accum_iter:-1}
world_size=${world_size:-"1"}
test_interval=${test_interval:-1}
drop_path=${drop_path:-"0.1"}

data=${data:-'placeholder'}
dataset=${dataset:-'imagenet'}
arch=${arch:-'vit_small'}
lr=${lr:-"5e-4"}
layer_decay=${layer_decay:-"0.65"}
wd=${wd:-"0.05"}
pretrain=${pretrain:-"None"}
pretrain_name=${pretrain_name:-"None"}
save_dir=${save_dir:-"checkpoints_tune"}
eval_gradient=${eval_gradient:-"False"}
eval_gradient_eval_aug=${eval_gradient_eval_aug:-"False"}
fixTo=${fixTo:-"-1"}
fixBnStat=${fixBnStat:-"False"}
add1BlockTo=${add1BlockTo:-""}
add1BlockToNumLayers=${add1BlockToNumLayers:-"1"}
reset_layers=${reset_layers:-"-1"}
add_batch_norm=${add_batch_norm:-"False"}

customSplit=${customSplit:-""}
customSplitName=${customSplitName:-""}
tuneFromFirstFC=${tuneFromFirstFC:-"False"}
eval_avg_attn_dist=${eval_avg_attn_dist:-"False"}

fc_scale=${fc_scale:-"1"}

no_aug=${no_aug:-"False"}
cutmix=${cutmix:-"1.0"}
mixup=${mixup:-"0.8"}
reprob=${reprob:-"0.25"}
aa=${aa:-""}

norm_beit=${norm_beit:-"False"}

avg_attn_dist_reg=${avg_attn_dist_reg:-"-1"}
avg_attn_dist_reg_target=${avg_attn_dist_reg_target:-""}
avg_attn_dist_reg_target_name=${avg_attn_dist_reg_target_name:-""}

resume=${resume:-"None"}
msr=${msr:-"False"}


exp="Mae_lr${lr}wd${wd}LayerD${layer_decay}B${batch_size}E${epochs}_${dataset}"

if [[ ${drop_path} != "0.1" ]]
then
  exp=${exp}_dropP${drop_path}
fi

if [[ ${accum_iter} != 1 ]]
then
  exp=${exp}_accuIter${accum_iter}
fi

if [[ ${customSplit} != "" ]]
then
  exp=${exp}_${customSplitName}
fi

if [[ ${tuneFromFirstFC} == "True" ]]
then
  exp=${exp}_tuneFromFirstFC
fi

if [[ ${no_aug} == "True" ]]
then
  exp=${exp}_noAug
fi

if [[ ${cutmix} != "1.0" ]]
then
  exp=${exp}_cutMix${cutmix}
fi

if [[ ${mixup} != "0.8" ]]
then
  exp=${exp}_mixup${mixup}
fi

if [[ ${reprob} != "0.25" ]]
then
  exp=${exp}_repro${reprob}
fi

if [[ ${aa} != "" ]]
then
  exp=${exp}_AA-${aa}
fi

if [[ ${norm_beit} == "True" ]]
then
  exp=${exp}_normBeit
fi

if [[ ${fc_scale} != "1" ]]
then
  exp=${exp}_fcScale${fc_scale}
fi

if [[ ${avg_attn_dist_reg} != "-1" ]]
then
  exp="${exp}_avgAttnR${avg_attn_dist_reg}Of${avg_attn_dist_reg_target_name}"
fi

if [[ ${pretrain} != "None" ]]
then
  exp=${exp}__${pretrain_name}
fi

if [[ ${eval_gradient} == "True" ]]
then
  exp=${exp}_evalGrad
  if [[ ${eval_gradient_eval_aug} == "True" ]]
  then
    exp=${exp}EvalAug
  fi
fi

if [[ ${eval_avg_attn_dist} == "True" ]]
then
  exp=${exp}_evalAvgAttnDist
fi

if [[ ${fixTo} != "-1" ]]
then
  exp=${exp}_fixTo${fixTo}
fi

if [[ ${fixBnStat} == "True" ]]
then
  exp=${exp}_fixBnStat
fi

if [[ ${add1BlockTo} != "" ]]
then
  exp=${exp}_add1BlockTo${add1BlockTo}
  if [[ ${add1BlockToNumLayers} != "1" ]]
  then
    exp=${exp}layersN${add1BlockToNumLayers}
  fi
fi

if [[ ${reset_layers} != "-1" ]]
then
  exp=${exp}_resetLayers${reset_layers}
fi

if [[ ${add_batch_norm} == "True" ]]
then
  exp=${exp}_bnFc
fi

if [[ ${msr} == "True" ]]
then
  launch_cmd="python"
else
  launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"
fi


cmd="${launch_cmd} main_finetune_mae.py --arch ${arch} --blr ${lr} --weight_decay ${wd} --epochs ${epochs} \
--batch_size ${batch_size} --drop_path ${drop_path} --reprob ${reprob} --mixup ${mixup} --cutmix ${cutmix} --dist_eval --layer_decay ${layer_decay} \
--data ${data} --dataset ${dataset} --num_workers ${workers} --output_dir ${save_dir}/${exp} \
--eval_interval ${test_interval} --accum_iter ${accum_iter} --fc_scale ${fc_scale} --world_size ${world_size}"

if [[ ${msr} == "True" ]]
then
  cmd="${cmd} --dist_on_itp"
fi

if [[ ${aa} != "" ]]
then
  cmd="${cmd} --aa ${aa}"
fi

if [[ ${norm_beit} != "" ]]
then
  cmd="${cmd} --norm_beit"
fi

if [[ ${fixTo} != "-1" ]]
then
  cmd="${cmd} --fixTo ${fixTo}"
fi

if [[ ${fixBnStat} == "True" ]]
then
  cmd="${cmd} --fixBnStat"
fi

if [[ ${add1BlockTo} != "" ]]
then
  cmd="${cmd} --add1BlockTo ${add1BlockTo} --add1BlockToNumLayers ${add1BlockToNumLayers}"
fi

if [[ ${reset_layers} != "-1" ]]
then
  cmd="${cmd} --reset_layers ${reset_layers}"
fi

if [[ ${add_batch_norm} == "True" ]]
then
  cmd="${cmd} --add_batch_norm"
fi

if [[ ${pretrain} != "None" ]]
then
  cmd="${cmd} --finetune ${pretrain}"
fi

if [[ ${avg_attn_dist_reg} != "-1" ]]
then
  cmd="${cmd} --avg_attn_dist_reg ${avg_attn_dist_reg} --avg_attn_dist_reg_target ${avg_attn_dist_reg_target}"
fi

if [[ ${eval_gradient} == "True" ]]
then
  cmd="${cmd} --eval_gradient"
  if [[ ${eval_gradient_eval_aug} == "True" ]]
  then
    cmd="${cmd} --eval_gradient_eval_aug"
  fi
fi

if [[ ${eval_avg_attn_dist} == "True" ]]
then
  cmd="${cmd} --eval_avg_attn_dist"
fi

if [[ ${no_aug} == "True" ]]
then
  cmd="${cmd} --no_aug"
fi

if [[ ${customSplit} != "" ]]
then
  cmd="${cmd} --customSplit ${customSplit}"
fi

if [[ ${tuneFromFirstFC} == "True" ]]
then
  cmd="${cmd} --tuneFromFirstFC"
fi

if [[ ${resume} != "None" ]]
then
  if [[ ${resume} == "local" ]]
  then
    resume=${save_dir}/${exp}/checkpoint_last.pth
  fi
  cmd="${cmd} --resume ${resume}"
fi

echo ${cmd}
${cmd}
