# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:34:03 2021

@author: vinograd
"""
             
from argparse import ArgumentParser
import json
from datetime import datetime, timedelta
from aberration.model import PhaseNet, Config, Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import keras.backend as K
K.clear_session()

parser = ArgumentParser()
parser.add_argument("-z", "--zernikes", dest="zern", help="list zernike modes to synthesize and to train on") #, type=list)
parser.add_argument("-ep","--train_epochs",dest="train_epochs", type=int)
parser.add_argument("-ph", "--phantom_filepath", dest ="phantom_filepath",
                    help="path to phantom(s) [str or list of strs]. none (default) for training on synthetic points",
                    default='none')

parser.add_argument("-b", "--train_batch_size",dest="train_batch_size",default=24, type=int)
parser.add_argument("-steps","--train_steps_per_epoch",dest="train_steps_per_epoch",default=24, type=int)
parser.add_argument("--crop_shape", dest="crop_shape", default=32, type=int)
parser.add_argument("--jitter", dest="jitter", default=0, type = int, #False, type=bool,
                    help="int: 0 = false, 1 = true = randomly move the center point within a given limit")
parser.add_argument("--max_jitter", dest = "max_jitter", default=None, help="series in z,y,x")

parser.add_argument("--outfolder",dest="outfolder", default='',
                    help="folder to save the model & plots")

parser.add_argument("-lr", "--learning_rate",dest="learning_rate", default=0.0003, type=float)

parser.add_argument("--range", dest="range", default=0.1, type=float,
                help="Training amplitude range. It was 0.15 until 21.03.22")

parser.add_argument("--noise_mean", dest="noise_mean", default=None,type=str, help="Pass a list with commas or a single int value, e.g. 70,130")
parser.add_argument("--noise_snr", dest="noise_snr", default=None,type=str)
parser.add_argument("--noise_sigma", dest="noise_sigma", default=None,type=str)
#parser.add_argument("--noise_perlin_flag", dest="noise_perlin_flag", default=False)
parser.add_argument("--gaussian_blur_sigma", dest="gaussian_blur_sigma", default=None, type=str)
parser.add_argument("--zernike_order", dest="zernike_order", default='ansi',type=str)
parser.add_argument("-na","--psf_na_detection", dest="psf_na_detection", default=1.4, type=float)

parser.add_argument("-psf_lam", dest="psf_lam", default=0.568, type=float)#,                   help="0.568 nm for red light, 0.488 for green light from Lucifer Yellow")

parser.add_argument("-unitsxy","--psf_unitsxy", dest="psf_unitsxy", default=0.094872, type=float)
parser.add_argument("-unitsz","--psf_unitsz", dest="psf_unitsz", default=0.2, type=float)
parser.add_argument("--psf_n", dest="psf_n", default=1.5, type=float)

#parser.add_argument("--planes",dest="planes", default=None, type=str,help="param planes: list, z planes with respect to center, if None then it takes all the planes")
#parser.add_argument("-initcrop", "--initial_crop_shape", dest="initcrop", #default=0, type=int,
#                    help="size of the crop manually cropped from the denoised microscopic image")
# Initcrop is useful only for training with "planes". Training on selected planes leads to worse results in general, therefore both are off for code simplicity.

args = parser.parse_args()

if args.noise_snr is None and args.noise_sigma is not None and args.noise_mean is not None:
    args.noise_snr = args.noise_mean/args.noise_sigma
    print("snr set to: ", args.noise_snr)
"""
if args.initcrop == 0:
    args.initcrop = None
else:
    args.initcrop = (args.initcrop,args.initcrop,args.initcrop)
"""

args.jitter = bool(args.jitter)
print("jitter = ", args.jitter)
print("zernike order = ",args.zernike_order)

print("args.noise_mean, noise_sigma, noise_snr, gaussian_blur: ", args.noise_mean, args.noise_sigma, args.noise_snr, args.gaussian_blur_sigma )

if args.max_jitter is not None:
        max_jitter_str = (args.max_jitter).strip("[ ]").replace(" ","").split(",")
        max_jitter = []
        for el in max_jitter_str:
                max_jitter.append(int(el))
        args.max_jitter = max_jitter
if args.noise_mean is not None:
        noise_mean_str = (args.noise_mean).strip("[ ]").replace(" ","").split(",")
        noise_mean = []
        for el in noise_mean_str:
                noise_mean.append(float(el))
        args.noise_mean = noise_mean
if args.noise_snr is not None:
        noise_snr_str = (args.noise_snr).strip("[ ]").replace(" ","").split(",")
        noise_snr = []
        for el in noise_snr_str:
                noise_snr.append(float(el))
        args.noise_snr = noise_snr
if args.noise_sigma is not None:
        noise_sigma_str = (args.noise_sigma).strip("[ ]").replace(" ","").split(",")
        noise_sigma = []
        for el in noise_sigma_str:
                noise_sigma.append(float(el))
        args.noise_sigma = noise_sigma
if args.gaussian_blur_sigma is not None:
        gaussian_blur_sigma_str = (args.gaussian_blur_sigma).strip("[ ]").replace(" ","").split(",")
        gaussian_blur_sigma = []
        for el in gaussian_blur_sigma_str:
                gaussian_blur_sigma.append(float(el))
        args.gaussian_blur_sigma = gaussian_blur_sigma

print("arg.max_jitter, args.noise_mean, noise_sigma, noise_snr, gaussian_blur: ", args.max_jitter,args.noise_mean, args.noise_sigma, args.noise_snr, args.gaussian_blur_sigma )


#args.planes = [-1,0,1]
"""print("args.planes: ",args.planes)
if args.planes is not None:
	planes_str = (args.planes).strip("[ ]").replace(" ","").split(",")
	planes = []
	for el in planes_str:
		planes.append(int(el))
else:
	planes=None
print("planes: ", planes)
"""
planes = None
              
zern_str = str(args.zern)#.strip("[, ]").replace(",","").replace(" ","")
print(zern_str)
zern = zern_str.split(",")
zern_str = ''.join(zern)
print(zern_str)
#zern = args.zern #list(zern_str) 



print("zern = ", zern, "  zern_str = ", zern_str)
print("types: zern ",type(zern), "zern_str ", type(zern_str) )
timestamp = datetime.now()
timestr = timestamp.strftime("%d%m_%H%M")
if args.outfolder == '':
    #args.outfolder = Path("./phasenet-extension/") / f"Zern{args.zern}_{timestr}//"
    args.outfolder = Path(f"./results/Zern{zern_str}_{timestr}/")
    args.outfolder.mkdir(exist_ok=True, parents=True)
    args.outfolder = str(args.outfolder)+'/'
print("       outfolder =: ", args.outfolder)

amp_range = [args.range]*len(zern)
amps = dict(zip(zern, amp_range))


print("Microscope parameters: ", args.psf_na_detection, " (",args.psf_unitsz,args.psf_unitsxy,args.psf_unitsxy,") ", args.psf_n,args.psf_lam)
if args.phantom_filepath == "none":
    print("\n\n     NO Phantom:       :",args.phantom_filepath)
    c = Config(zernike_amplitude_ranges=amps, psf_na_detection=args.psf_na_detection, psf_units=(args.psf_unitsz,args.psf_unitsxy,args.psf_unitsxy), psf_n=args.psf_n,
           psf_lam_detection=args.psf_lam, noise_mean=args.noise_mean, noise_snr=args.noise_snr, noise_sigma=args.noise_sigma, zernike_order=args.zernike_order,
           gaussian_blur_sigma=args.gaussian_blur_sigma,
           train_batch_size=args.train_batch_size,
           train_epochs=args.train_epochs,
           train_steps_per_epoch=args.train_steps_per_epoch,
           crop_shape=(args.crop_shape,args.crop_shape,args.crop_shape),
           #initcrop=args.initcrop,
           jitter=args.jitter, max_jitter=args.max_jitter,
         train_learning_rate= args.learning_rate #default 0.0003
        , planes = planes
	)
else:
    print("\n\n     Phantom:       :",args.phantom_filepath)     
    c = Config(zernike_amplitude_ranges=amps, psf_na_detection=args.psf_na_detection, psf_units=(args.psf_unitsz,args.psf_unitsxy,args.psf_unitsxy), psf_n=args.psf_n,
           psf_lam_detection=args.psf_lam, noise_mean=args.noise_mean, noise_snr=args.noise_snr, noise_sigma=args.noise_sigma, zernike_order=args.zernike_order,
           gaussian_blur_sigma=args.gaussian_blur_sigma,
           train_batch_size=args.train_batch_size,
           train_epochs=args.train_epochs,
           train_steps_per_epoch=args.train_steps_per_epoch,
           crop_shape=(args.crop_shape,args.crop_shape,args.crop_shape),
           #initcrop=args.initcrop,
           jitter=args.jitter, max_jitter=args.max_jitter,
           phantom_params = {'name':'images','filepath':args.phantom_filepath},
	 train_learning_rate= args.learning_rate #default 0.0003
	, planes = planes
	)

# safe config to txt
with open(args.outfolder+'config.txt', 'w+') as file:
     file.write(json.dumps(vars(c))) # use `json.loads` to do the reverse
with open(args.outfolder+'commandline_args.txt', 'w+') as f:
    json.dump(args.__dict__, f, indent=2)
#encode: 'Zern489', 'crop64', 'in64','on8'


code_name=f'{c.train_epochs}ep_cellimage'
model = PhaseNet(config=c, name=code_name, basedir=args.outfolder)
history =model.train()

plt.figure(figsize=(10,8))
plt.plot(history.history['loss'], marker='*', label='loss')
plt.plot(history.history['val_loss'], '--', marker='*', label='val_loss')
plt.xlabel('epoch')
plt.ylabel('log loss')
plt.yscale('log')
plt.legend()
plt.title(f'{args.range} Range \n Zernike {zern}\n crop{args.crop_shape} Phantom {c.train_epochs} epochs')
plt.savefig(args.outfolder+'loss_'+code_name+'.png')

## Validation 
data = Data(
    batch_size           = 50,
    amplitude_ranges     = model.config.zernike_amplitude_ranges,
    order                = model.config.zernike_order,
    normed               = model.config.zernike_normed,
    psf_shape            = model.config.psf_shape,
    units                = model.config.psf_units,
    na_detection         = model.config.psf_na_detection,
    lam_detection        = model.config.psf_lam_detection,
    n                    = model.config.psf_n,
    noise_mean           = model.config.noise_mean,
    noise_snr            = model.config.noise_snr,
    noise_sigma          = model.config.noise_sigma,
    noise_perlin_flag    = model.config.noise_perlin_flag,
    crop_shape           = model.config.crop_shape,
    jitter               = model.config.jitter,
    max_jitter           = model.config.max_jitter,
    phantom_params       = model.config.phantom_params,
    planes               = model.config.planes,
)
psfs, amps = next(data.generator())
amps_pred = np.array([model.predict(psf) for psf in tqdm(psfs)])

for zernike_ind in range(len(zern)):
    zernike = tuple(model.config.zernike_amplitude_ranges.keys())[zernike_ind]
    plt.figure(figsize=(10,8))
    ind = np.argsort(amps[...,zernike_ind]) #.ravel())
    plt.plot(amps[ind,zernike_ind], marker='*', label='gt')
    plt.plot(amps_pred[ind,zernike_ind], '--', marker='*', label='pred')
    plt.xlabel('test psf')
    plt.ylabel(f'amplitude')
    plt.legend()
    #plt.title(f'Zernike {zernike} \ntrain on {zern} Zernike\n crop{args.crop_shape} (train {args.initcrop}) {c.train_epochs} epochs {args.range} Range') #\nphantom=raw_neurons_crop64/N_0_000_crop64
    plt.title(f'Zernike {zernike} \ntrain on {zern} Zernike\n crop{args.crop_shape} (train {c.train_epochs} epochs {args.range} Range')
    plt.savefig(args.outfolder+f'syntest_z'+str(zernike)+'_'+code_name+'.png')
    

K.clear_session()

