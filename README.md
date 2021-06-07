# headless-crawlingathome

This script runs https://github.com/TheoCoombes/crawlingathome from linux box. SVG images will most likely be lost in this scenario. Otherwise the script will behave very similar to its Colab source at: https://colab.research.google.com/drive/1P1H-1kc_CFgJE1NOnXywm2fVSoyv2gMW

## How to use on your home computer

1. create an empty environment with python 3.8
2. make sure to increase the max number of open files with this command:```ulimit -n 65536``` and confirm the new limits with ```ulimit -n```
3. run ```git clone https://github.com/rvencu/headless-crawlingathome```
4. run ```python ./headless-crawlingathome/setup.py```
5. edit ```./headless-crawlingathome/test.py``` to edit your nickname on line 2
6. run ```python ./headless-crawlingathome/test.py``` to contribute

## Droplet deployment in cloud

Very cheap droplets can be employed to crawl. For instance the CX11 from https://www.hetzner.com/cloud?country=ro is only 3 Euro per month and it can process about 500 shards in one month

1. at this time only the cloud config version 2 works. You need to use the content of the ```cloud-config-2.yaml``` as cloud-init script to spin up a new droplet.
2. inside the script change your public ssh key at line 10 and your nickname at lines 54 and 55 (just replace ```yournick``` with your nickname)
3. spin up the droplet (min 1vCPU, 20GB disk) and wait for the setup and reboot
4. the droplet will run on itself. Usually a shard is done in about 5000 seconds, similar to google colab without GPU backend
5. optionally monitor the script like this: ssh into droplet with username ```crawl``` and your publickey and launch ```tail -f crawl.log```

## Upgrading procedure

1. the safest way is to erase the local clone with commands like ```rm -R /path/to/working/dir```
2. use the above instructions to reinstall. the setup.py will take care to clone the up to date repos for the client, clip, etc.

### todo
test.py script still needs optimization

No more Google Colab kickouts...
