import numpy as np
import argparse, os, re, librosa
from skimage.transform import resize

from nsgt3.cq import NSGT
from nsgt3.fscale import Scale, OctScale, LogScale, MelScale

VALID_EXTS=['.wav', '.aif', '.aiff']
SCALE_DICT = {'oct':OctScale, 'log':LogScale, 'mel':MelScale}


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help="folder containing audio files", type=str)
parser.add_argument('-o', '--output', help="destination path", type=str)
parser.add_argument('-s', '--scale', help="scale for nsgt", choices=['oct', 'log', 'mel'], default='oct')
parser.add_argument('-r', '--resample', type=int, help="forces resampling of incoming data", default=22050)
parser.add_argument('-d', '--downsample', type=int, help="downsample factor", default=20)
parser.add_argument('--min_freq', type=int, help="starting frequency for frequency scale", default=30)
parser.add_argument('--max_freq', type=int, help="ending frequency for frequency scale", default=11000)
parser.add_argument('--bins', type=int, help="ending frequency for frequency scale", default=48)


args = parser.parse_args()

input_files = []
output_files = []

assert os.path.exists(args.input), "input location %s does not exist"%args.input
if not os.path.exists(args.output):
    os.makedirs(args.output)
for root, directory, files in os.walk(args.input):
    valid_files = list(filter(lambda x: os.path.splitext(x)[1] in VALID_EXTS, files))
    full_paths = [root+'/'+f for f in valid_files]
    input_files.extend(full_paths)
    output_files.extend([os.path.splitext(re.sub(args.input, args.output, f))[0]+'.npy' for f in full_paths])


scl = SCALE_DICT[args.scale](args.min_freq, args.max_freq, args.bins)

for i in range(len(input_files)):
    y, sr = librosa.load(input_files[i])
    y = librosa.core.resample(y, sr, args.resample)
    nsgt = NSGT(scl, args.resample, len(y), real=True, matrixform=True, reducedform=1)
    transform = np.array(list(nsgt.forward(y))).T
    transform_abs = np.abs(transform); transform_angle = np.angle(transform);
    transform_abs = resize(transform_abs, (int(transform.shape[0] / args.downsample), transform.shape[1]), mode='constant')
    transform_angle = resize(transform_angle, (int(transform.shape[0] / args.downsample), transform.shape[1]), mode='constant')
    transform = transform_abs*np.exp(1j*transform_angle)

    current_output_file = output_files[i]
    current_dir = os.path.dirname(current_output_file);
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    np.save(output_files[i], transform)

    
