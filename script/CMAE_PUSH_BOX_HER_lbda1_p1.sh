#!/usr/bin/env bash
cd ..
i=0
EXP_BATCH_NAME=CMAE
EXP=CMAE_PUSH_BOX_HER_lbda1_p1
ENV=push_box
python main.py --env push_box --exp_mode active_dqn --multilevel --tree_subspace  --mixed_explore --stochastic_select_subspace --alpha1 0.1 --alpha2 0.05  --exp_name ${EXP} --seed ${i} --exp_batch_name ${EXP_BATCH_NAME} --lbda 1 --p 1 --use_HER True



