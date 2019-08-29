# USIDEU HMR Project

### Script Guide

#### Making HD stick figure and mesh overlay from imgdirectory
prepare frames in the directory in `png` format then run <br>

`python -m ufit-imgdir --img_path /home/ubuntu/s3/190802_demo/short/demo2_frames/`

to convert the result to video with highest quality try: <br>
`ffmpeg -i wireframe/frame_%6d.png -b:v 8M demo2_wireframe.mp4`

### Installation guide

On Ubuntu 16.04

* installing llvm3.9 with (needed for mesa) <br>
`sudo apt-get install clang-3.9 lldb-3.9` <br>
* installing bison and flex <br>
`sudo apt-get install bison flex` <br>
* installing zlib <br>
`sudo apt-get install zlib1g-dev` <br>
* installing glproto<br>
`sudo apt-get install x11proto-gl-dev`<br>
* installing libdrm<br>
`sudo apt-get install libdrm-dev`<br>
* installing dri2proto<br>
`sudo apt-get install x11proto-dri2-dev`<br>
* installing x libs: <br>
`sudo apt-get install xorg-dev libxcb* libx11-xcb-dev libxshmfence-dev`<br>
* installing libelf<br>
`sudo apt-get install libelf-dev`<br>

* downloading mesa 19.06 and installing mesa: (alternatively you could do `sudo apt-get install libosmesa6-dev` and `libglu1-mesa-dev`)

` ./configure --enable-autotools --enable-llvm ac_cv_path_LLVM_CONFIG=/usr/lib/llvm-3.9/bin/llvm-config --enable-osmesa`

* make mesa
`make -j8`

* install mesa
`sudo make install`

* intall anaconda python 3.7<br>

* installing latest nvidia drivers ( if the box has secure boot feature enabled it otherwise nvidia drivers will mess up the system you can't login into GUI desktop environment (a bug)). 

```
sudo apt-get purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa and then sudo apt-get update
sudo apt-get install nvidia-384 nvidia-modprob
```
* then fix the path in `.bashrc` and append: 
`export PATH=$PATH$:/home/alireza/anaconda3/bin:/usr/lib/nvidia-384/bin/`<br>
and run `source ~/.bashrc`

* Now creating a pytorch conda environment to make sure pytorch sees the GPU and CUDA drivers: <br>
`conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`
* switch to environment and test
```
source activate pytorch_1.1
python -c 'import torch;print(torch.cuda.is_available())'
```
* test tensorflow's access to gpu
```
# on python command line
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

* for XLA CPU set environment variable as follows 
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
```

### Troubleshooting Nvidia driver and Graphical Desktop login with Secure boot
On systems with secure boot capability the secure boot has to be disabled for nvidia driver to work and be recognized. But typically after installing nvidia driver the desktop can't login and the xserver doesn't work propertly due to some permission issues (probably with .ICEauthority and .Xauthority). The solutions on the internet has not worked for. (deleting the two files or chown to current user and group and making sure the /tmp is writable, by all users). I tried switching to xdm from lightdm. after doing `sudo apt-get install xdm` and deleting the two files the desktop loopedback once and then for second time let me login. 



# ORIGINAL README End-to-end Recovery of Human Shape and Pose

Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik
CVPR 2018

[Project Page](https://akanazawa.github.io/hmr/)
![Teaser Image](https://akanazawa.github.io/hmr/resources/images/teaser.png)

### Requirements
- Python 2.7
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.3, demo alone runs with TF 1.12

### Installation

#### Linux Setup with virtualenv
```
virtualenv venv_hmr
source venv_hmr/bin/activate
pip install -U pip
deactivate
source venv_hmr/bin/activate
pip install -r requirements.txt
```
#### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==1.3.0
```
Without GPU:
```
pip install tensorflow==1.3.0
```

### Windows Setup with python 3 and Anaconda
This is only partialy tested.
```
conda env create -f hmr.yml
```
#### if you need to get chumpy 
https://github.com/mattloper/chumpy/tree/db6eaf8c93eb5ae571eb054575fb6ecec62fd86d


### Demo

1. Download the pre-trained models
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz
```

2. Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```

Images should be tightly cropped, where the height of the person is roughly 150px.
On images that are not tightly cropped, you can run
[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and supply
its output json (run it with `--write_json` option).
When json_path is specified, the demo will compute the right scale and bbox center to run HMR:
```
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
```
(The demo only runs on the most confident bounding box, see `src/util/openpose.py:get_bbox`)

### Webcam Demo (thanks @JulesDoe!)
1. Download pre-trained models like above.
2. Run webcam Demo
2. Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```

### Training code/data
Please see the [doc/train.md](https://github.com/akanazawa/hmr/blob/master/doc/train.md)!

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}
```

### Opensource contributions
[Dawars](https://github.com/Dawars) has created a docker image for this project: https://hub.docker.com/r/dawars/hmr/

[MandyMo](https://github.com/MandyMo) has implemented a pytorch version of the repo: https://github.com/MandyMo/pytorch_HMR.git

[Dene33](https://github.com/Dene33) has made a .ipynb for Google Colab that takes video as input and returns .bvh animation!
https://github.com/Dene33/video_to_bvh 

<img alt="bvh" src="https://i.imgur.com/QxML83b.gif" /><img alt="" src="https://i.imgur.com/vfge7DS.gif" />
<img alt="bvh2" src=https://i.imgur.com/UvBM1gv.gif />

I have not tested them, but the contributions are super cool! Thank you!!


