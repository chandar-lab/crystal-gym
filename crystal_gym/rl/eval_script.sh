indices=(3403 630 2271 8906)
for ind in ${indices[@]}
do
exp_name=ppo-diff-small-band_gap-$ind-2.0-0.01-0.5
total_timesteps=25
num_steps=5
for seed in 10 20 30
    do

        # python ppo.py wandb.mode=offline exp.exp_name=$exp_name exp.seed=$seed env.seed=$seed env.index=3403 env.property=bm env.p_hat=500.0 qe.tstress=true qe.tprnfor=true qe.occupations=smearing algo.ent_coef=>
        
    if [[ $ind -eq 2271 ]]; then
        total_timesteps=40
        num_steps=8
    elif [[ $ind -eq 8906 ]]; then
        total_timesteps=20
        num_steps=4
    fi


    python ppo.py wandb.mode=offline exp.exp_name=$exp_name exp.seed=$seed env.seed=$seed env.index=$ind env.property=band_gap env.p_hat=2.0 qe.tstress=false qe.tprnfor=false qe.occupations=fixed algo.ent_coef=0.0>

    #    python rainbow.py wandb.mode=offline \
    #        algo.total_timesteps=$total_timesteps \
    #        exp.mode=eval \
    #        exp.exp_name=$exp_name \
    #        exp.seed=$seed \
    #        env.seed=$seed \
    #        env.index=$ind \
    #        env.property=density \
    #       env.p_hat=5.0 \
    #        qe.tstress=false \
    #        qe.tprnfor=false \
    #        qe.occupations=smearing \
    #        algo.buffer_size=2000 \
    #        algo.target_network_frequency=100 \
    #        qe.calculation=vc-relax \
    #        env.vocab=small \
    #        algo.multi_step=3
    done
    done