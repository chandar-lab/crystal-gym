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
        for ent_coef in 0.01
            do
            for vf_coef in 0.5 #1.0
            do
            for ind in 8666 2271 630 3403 2190 8906 8354
                do 
                    for seed in 10 20 30 #40 50
                        do
                            sbatch --job-name="PPO-NEW-${prop^^}-$ind" script CRYSTALGYM PPO-$prop-$ind ppo-medium-$prop-$ind-$p_hat-$ent_coef-$vf_coef $seed $seed $ind $prop $p_hat $tstress $tprnfor $occupations $ent_coef $vf_coef
                        done
                done
            done
            done
    done 
