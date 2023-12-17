# test Obama eo
method=r2talker
cond_type=idexp
vid=Obama

python test.py \
    --pose data/${vid}/transforms_val.json \
    --ckpt ./pretrained/${r2talker}_${vid}_${cond_type}_torso.pth \
    --aud data/${vid}/intro_eo.npy \
    --workspace trial_test \
    --bg_img data/${vid}/bc.jpg \
    -O --torso --data_range 0 100 --preload 2

# merge audio with video
ffmpeg -y -i trial_test/results/${r2talker}_${vid}_${cond_type}.mp4 -i data/intro.wav -c:v copy -c:a aac ${r2talker}_${vid}_${cond_type}_aud.mp4
