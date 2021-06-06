# headless-crawlingathome

This script runs https://github.com/TheoCoombes/crawlingathome from linux box. SVG images will most likely be lost in this scenario. Otherwise the script will behave very similar to its Colab source at: https://colab.research.google.com/drive/1P1H-1kc_CFgJE1NOnXywm2fVSoyv2gMW

## How to use

1. create an empty environment with python 3.8
2. make sure to increase the max number of open files with this command:```ulimit -n 65536``` and confirm the new limits with ```ulimit -n```
3. run ```git clone https://github.com/rvencu/headless-crawlingathome```
4. run ```python ./headless-crawlingathome/setup.py```
5. edit ```./headless-crawlingathome/test.py``` to edit your nickname on line 2
6. run ```python ./headless-crawlingathome/test.py``` to contribute

For deployment to 1vCPU, 1GB RAM droplet, replace with ```./headless-crawlingathome/crawlfulltest.py``` at steps 5 and 6. In addition to the above setup, the droplet should be configured with 5GB swap file to fit requirements for pytorch and CLIP

## Upgrading procedure

1. the safest way is to erase the local clone with commands like ```rm -R /path/to/working/dir```
2. use the above instructions to reinstall. the setup.py will take care to clone the up to date repos for the client, clip, etc.

### todo
test.py script still needs optimization

No more Google Colab kickouts...
