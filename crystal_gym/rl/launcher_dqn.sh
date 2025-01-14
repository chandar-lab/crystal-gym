#!/bin/bash


for prop in density bm band_gap
    do
        if [ $prop == "bm" ]
        then
            p_hat=300.0
            tstress=true
            tprnfor=true
            occupations=smearing
            calculation=scf
            tsteps=100000
        elif [ $prop == "band_gap" ]
        then
            p_hat=1.12
            tstress=false
            tprnfor=false
            occupations=fixed
            calculation=scf
            tsteps=200000
        elif [ $prop == "density" ]
        then
            p_hat=3.0
            tstress=false
            tprnfor=false
            occupations=smearing
            calculation=vc-relax
            tsteps=100000
        fi
        for ind in 8666 2271 630 3403 2190 8906 8354
        do
            for rb_size in 2000 #10000
                do
                for target_freq in 100 #500
                    do
                        for seed in 10 20 30
                        do
                            # dqn-priority-$prop-$ind-$rb_size-$target_freq
                            # dqn-correct-$prop-$ind-$rb_size-$target_freq
                            sbatch --job-name="DQN-HP-${prop^^}-$ind-$rb_size-$target_freq" script.sh CRYSTALGYM DQN-$prop-$ind-$rb_size-$target_freq dqn-smallfin-$prop-$p_hat-$ind-$rb_size-$target_freq-$tsteps $seed $seed $ind $prop $p_hat $tstress $tprnfor $occupations $rb_size $target_freq $calculation $tsteps
                        done
                    done
                done
        done
    done
