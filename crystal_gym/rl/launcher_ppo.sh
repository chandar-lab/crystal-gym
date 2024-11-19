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
        for ind in 8666 2271 #630 3403 2190 8906 8354
            do 
                for seed in 10 20 30
                do
                    sbatch --constraint ampere --job-name="PPO-NEW-${prop^^}-$ind" script PPO PPO-$prop-$ind ppo-basic-$prop-$ind-$p_hat $seed $seed $ind $prop $p_hat $tstress $tprnfor $occupations
                done
            done
    done 
