import argparse

from nerf_triplane.provider import NeRFDataset
from torch.cuda import is_available
import time 

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
parser.add_argument('--workspace', type=str, default='workspace')
parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=200000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='latest')
parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

### loss set
parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")
parser.add_argument('--pyramid_loss', type=int, default=0, help="use perceptual loss")

### network backbone options
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

parser.add_argument('--bg_img', type=str, default='', help="background image")
parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")
parser.add_argument('--bs_area', type=str, default="upper", help="upper or eye")
parser.add_argument('--au45', action='store_true', help="use openface au45")
parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

### dataset options
parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
# (the default value is for the fox dataset)
parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

parser.add_argument('--init_lips', action='store_true', help="init lips region")
parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

parser.add_argument('--torso', action='store_true', help="fix head and train torso")
parser.add_argument('--head_ckpt', type=str, default='', help="head model")

### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=450, help="GUI width")
parser.add_argument('--H', type=int, default=450, help="GUI height")
parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

### else
parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")
parser.add_argument('--portrait', action='store_true', help="only render face")
parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
parser.add_argument('--ind_num', type=int, default=20000, help="number of individual codes, should be larger than training dataset size")

parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

# asr
parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
parser.add_argument('--asr_play', action='store_true', help="play out the audio")

parser.add_argument('--asr_model', type=str, default='deepspeech')

parser.add_argument('--asr_save_feats', action='store_true')
# audio FPS
parser.add_argument('--fps', type=int, default=50)
# sliding window left-middle-right length (unit: 20ms)
parser.add_argument('-l', type=int, default=10)
parser.add_argument('-m', type=int, default=50)
parser.add_argument('-r', type=int, default=10)
parser.add_argument('--quantize', type=int, default=1, choices=[0,1], required=False)
parser.add_argument('--log-path', type=str, default='quantize_log.csv')
# https://pytorch.org/docs/stable/quantization.html#quantized-model
parser.add_argument('--quantize-type', type=str, default='qint8', choices=['float16', 'qint8'], required=False)
opt = parser.parse_args()

device = 'cuda' if is_available() else 'cpu'
test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

curr_batch = 0
max_batch = 1000

start = time.time()
for i, batch in enumerate(test_loader):
    curr_batch += 1
    if curr_batch%100 == 0:
        print(f'Loading batch {curr_batch}')

    if curr_batch == max_batch:
        break
    pass 

end = time.time()

avg_time = (end - start) / max_batch
with open('time_log.txt', 'a+') as f:
    f.write(f'Average time for each batch: {avg_time} from {opt.path}\n')

print(f'Took {avg_time}s to load single batch from {opt.path}')