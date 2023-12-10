# test May eo
python test.py \
    --pose data/May.json \
    --ckpt pretrained/May_eo.pth \
    --aud data/intro_eo.npy \
    --workspace trial_test \
    --bg_img data/bg.jpg \
    -O --torso --data_range 0 100 --preload 2

# merge audio with video
ffmpeg -y -i trial_test/results/ngp_ep0028.mp4 -i data/intro.wav -c:v copy -c:a aac May_eo_intro.mp4

# # test May ds
# python test.py \
#     --pose data/May.json \
#     --ckpt pretrained/May.pth \
#     --aud data/intro.npy \
#     --workspace trial_test \
#     --bg_img data/bg.jpg \
#     -O --torso --data_range 0 100 --asr_model deepspeech

# # merge audio with video
# ffmpeg -y -i trial_test/results/ngp_ep0056.mp4 -i data/intro.wav -c:v copy -c:a aac May_intro.mp4