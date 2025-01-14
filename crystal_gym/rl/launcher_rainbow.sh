#!/bin/bash


for prop in bm band_gap
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
        for ind in 630 8666 2271 3403 2190 8906 8354
        do
            for rb_size in 2000 #10000
                do
                for target_freq in 100 #500
                    do
                    for multi_step in 3
                    do
                        for seed in 10 20 30 #40 50
                        do
                            sbatch --job-name="RAINBOW-FIN-${prop^^}-$ind-$rb_size-$target_freq" script.sh CRYSTALGYM Rainbow-$prop-$ind-$rb_size-$target_freq-$multi_step Rainbow-smallfin-$prop-$ind-$p_hat-$rb_size-$target_freq-$multi_step-$tsteps $seed $seed $ind $prop $p_hat $tstress $tprnfor $occupations $rb_size $target_freq $multi_step $calculation $tsteps
                        done
                    done
                    done
                done
        done
    done
