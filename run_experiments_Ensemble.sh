#!/bin/bash

src_env="HalfCheetah-v2" # {InvertedPendulum-v2, Hopper-v2, HalfCheetah-v2}
trg_env="HalfCheetahBroken-v2" # {InvertedPendulumPositiveSkew-v2, HopperBroken-v2, HalfCheetahBroken-v2, HalfCheetahModified-v2}

# script to run grounding with five different random seeds
for ((i=1; i<2; i++))
do
  python3.7 ensemble_policy_learning.py \
  --src_env $src_env\
  --trg_env $trg_env\
  --rollout_set "MS" \
  --training_steps_policy 3000000\
  --namespace "CODE_release_test_script" \
  --num_atps 5 \
  --expt_number ${i}\
  --determinisitc_atp \
  --verbose 1 \
  --num_src 5 \
  --eval &
  wait
  echo
done
