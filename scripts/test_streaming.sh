# end-to-end test with audio streaming
python test.py \
    --pose data/May/transforms_train.json \
    --ckpt trial_May_eo_torso/checkpoints/ngp.pth \
    --aud data/intro_eo.npy \
    --workspace trial_test \
    --bg_img data/May/bc.jpg \
    -l 10 -m 10 -r 10 \
    -O --torso --data_range 0 100 --preload 2 --gui --asr