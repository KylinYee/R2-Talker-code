# test Obama r2talker
method=r2talker # r2talker, genefaceDagger, rad-nerf
cond_type=idexp # eo, idexp
vid=Obama

python test.py \
    --method ${method} \
    --cond_type ${cond_type} \
    --pose ./pretrained/transforms_val.json \
    --ckpt ./pretrained/${method}_${vid}_${cond_type}_torso.pth \
    --aud ./pretrained/test_lm3ds.npy \
    --workspace trial_test \
    --bg_img ./pretrained/bc.jpg \
    -O --torso --data_range 200 300 

# merge audio with video
ffmpeg -y -i trial_test/results/ngp_ep0028.mp4 -i ./pretrained/test.wav -c:v copy -c:a aac trial_test/results/${method}_${vid}_${cond_type}_aud.mp4


# method=genefaceDagger
# cond_type=idexp
# vid=Obama

# python test.py \
#     --method ${method} \
#     --cond_type ${cond_type} \
#     --pose data/${vid}/transforms_val.json \
#     --ckpt trial_genefaceDagger_Obama_idexp/checkpoints/ngp.pth \
#     --aud data/${vid}/aud_idexp_val.npy \
#     --workspace trial_test \
#     --bg_img ./pretrained/bc.jpg \
#     -O --torso --data_range 200 300 



# method=rad-nerf
# cond_type=eo
# vid=Obama

# python test.py \
#     --method ${method} \
#     --cond_type ${cond_type} \
#     --pose data/${vid}/transforms_val.json \
#     --ckpt trial_rad-nerf_Obama_eo_torso/checkpoints/ngp.pth \
#     --aud data/${vid}/aud_eo.npy \
#     --workspace trial_test \
#     --bg_img data/${vid}/bc.jpg \
#     -O --torso --data_range 200 300 