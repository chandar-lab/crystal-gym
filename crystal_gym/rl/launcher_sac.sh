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
        for ind in 630 2190 3403
            do 
                for seed in 10 20 30
                do
                    sbatch --constraint ampere --job-name="SAC-NEW-${prop^^}-$ind" script SAC SAC-$prop-$ind sac-basic-$prop-$ind $seed $seed $ind $prop $p_hat $tstress $tprnfor $occupations
                done
            done
    done 