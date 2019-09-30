#!/usr/bin/env bash

for i in 1 #2
do
    O="out_junk"
    C="check_junk"

    M='models_pretrained/ResNet18_cifar10.t7'
    MA='models_pretrained/ResNet18_cifar10_adv.t7'
    E=1
    EP=0
    LR=0.001

    echo `date`
    echo "starting run number $i"


    # finetuning only
    echo "Finetuning starting."

    python3 train_net.py --epochs $E --val_period 3 --model_path $MA --lr_schedule 3 8 --lr $LR --checkpoint_folder $C --output $O --adversarial
    python3 train_net.py --epochs $E --val_period 3 --model_path $M  --lr_schedule 3 8 --lr $LR --checkpoint_folder $C --output $O
    python3 train_net.py --epochs $E --val_period 3 --model_path $MA --rankmin --rank_sched 1 5 10 --lr_schedule 3 8 --lr $LR --checkpoint_folder $C --output $O --adversarial
    python3 train_net.py --epochs $E --val_period 3 --model_path $MA --rankmax --rank_sched 1 5 10 --lr_schedule 3 8 --lr $LR --checkpoint_folder $C --output $O --adversarial
    python3 train_net.py --epochs $E --val_period 3 --model_path $M --rankmax --rank_sched 1 5 10 --lr_schedule 3 8 --lr $LR --checkpoint_folder $C --output $O
    python3 train_net.py --epochs $E --val_period 3 --model_path $M --rankmin --rank_sched 1 5 10 --lr_schedule 3 8 --lr $LR --checkpoint_folder $C --output $O

    # Testing
    echo "Testing nets, output to table in $O."
    python3 test_net.py --model_path models_pretrained/ResNet18_cifar10.t7 --adversarial --output $O
    python3 test_net.py --model_path models_pretrained/ResNet18_cifar10_adv.t7 --adversarial --output $O
    python3 test_net.py --model_path $C/CIFAR10_minFalse_maxFalse_aFalse/epoch=$EP.t7 --adversarial --output $O
    python3 test_net.py --model_path $C/CIFAR10_minFalse_maxFalse_aTrue/epoch=$EP.t7 --adversarial --output $O
    python3 test_net.py --model_path $C/CIFAR10_minFalse_maxTrue_aFalse/epoch=$EP.t7 --adversarial --output $O
    python3 test_net.py --model_path $C/CIFAR10_minTrue_maxFalse_aFalse/epoch=$EP.t7 --adversarial --output $O
    python3 test_net.py --model_path $C/CIFAR10_minFalse_maxTrue_aTrue/epoch=$EP.t7 --adversarial --output $O
    python3 test_net.py --model_path $C/CIFAR10_minTrue_maxFalse_aTrue/epoch=$EP.t7 --adversarial --output $O

    # Plotting
    echo "removing previously computed singular values with command:"
    echo "rm -r ResNet18"
    rm -r ResNet18
    echo "Computing singular values."

    python3 singular_values.py --save_name Untrained
    python3 singular_values.py --model_path $C/CIFAR10_minFalse_maxFalse_aFalse/epoch=$EP.t7 --save_name Natural
    python3 singular_values.py --model_path $C/CIFAR10_minFalse_maxFalse_aTrue/epoch=$EP.t7 --save_name Adv
    python3 singular_values.py --model_path $C/CIFAR10_minFalse_maxTrue_aFalse/epoch=$EP.t7 --save_name RankMax
    python3 singular_values.py --model_path $C/CIFAR10_minTrue_maxFalse_aFalse/epoch=$EP.t7 --save_name RankMin
    python3 singular_values.py --model_path $C/CIFAR10_minFalse_maxTrue_aTrue/epoch=$EP.t7 --save_name RankMaxAdv
    python3 singular_values.py --model_path $C/CIFAR10_minTrue_maxFalse_aTrue/epoch=$EP.t7 --save_name RankMinAdv

    echo "Plotting."
    python3 plots.py --names Untrained Natural Adv RankMax RankMin RankMaxAdv RankMinAdv --output $O --pltnum 0
    python3 plots.py --names Untrained Adv RankMaxAdv RankMinAdv --output $O --pltnum 1
    python3 plots.py --names Untrained Natural RankMax RankMin --output $O --pltnum 2

    echo "ending run number $i"
    echo `date`
done