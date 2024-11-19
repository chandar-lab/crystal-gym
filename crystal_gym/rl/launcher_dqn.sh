#!/bin/bash


for prop in bm band_gap
    do
        if [ $prop == "bm" ]
        then
            p_hat=300.0
            tstress=true
            tprnfor=true
            occupations=smearing
        else
            p_hat=1.12
            tstress=false
            tprnfor=false
            occupations=fixed
        fi
        for ind in 630 3403 8666 8354
        do
            for rb_size in 2000 10000
                do
                for target_freq in 100 500
                    do
                        for seed in 10 20 30
                        do
                            sbatch --constraint ampere --job-name="DQN-HP-${prop^^}-$ind-$rb_size-$target_freq" script DQN DQN-$prop-$ind-$rb_size-$target_freq dqn-hp-$prop-$ind-$rb_size-$target_freq $seed $seed $ind $prop $p_hat $tstress $tprnfor $occupations $rb_size $target_freq
                        done
                    done
                done
        done
    done
