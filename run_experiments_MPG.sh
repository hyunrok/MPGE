#!/bin/bash

src_env="HalfCheetah-v2" # {InvertedPendulum-v2, Hopper-v2, HalfCheetah-v2}
trg_env="HalfCheetahBroken-v2" # {InvertedPendulumPositiveSkew-v2, HopperBroken-v2, HalfCheetahBroken-v2, HalfCheetahModified-v2}
# demo_sub_dir = {PositiveSkew1.5, BrokenHopper, BrokenCheetah, HeavyCheetah}

# script to run grounding with five different random seeds
for ((i=1; i<6; i++))
do
  python3.7 run_multi_policy_grounding.py \
  --src_env $src_env\
  --trg_env $trg_env\
  --rollout_set "MS" \
  --demo_sub_dir "BrokenCheetah" \
  --training_steps_atp 20000000\
  --training_steps_policy 3000000\
  --namespace "CODE_release_test_script" \
  --expt_number ${i}\
  --determinisitc_atp \
  --verbose 1 \
  --n-transitions 2000 \
  --num_src 5 \
  --tensorboard \
  --eval \
  --plot &
  wait
  echo
done

# --collect_demo