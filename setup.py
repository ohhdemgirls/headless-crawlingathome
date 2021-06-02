import os

os.system("pip install gevent")
os.system("pip install requests==2.25")

os.system("pip install tensorflow==2.4")
os.system("pip install pandas")

os.chdir('./headless-crawlingathome')
os.system("git clone https://github.com/TheoCoombes/crawlingathome")
os.system("pip install -r crawlingathome/requirements.txt")

os.system("pip install spacy==2.3")
os.system("python -m spacy download en_core_web_sm")

os.system("pip install IPython")

os.system("pip install grequests")

os.system("pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
")
os.system("pip install ftfy regex tqdm")
os.system("pip install git+https://github.com/openai/CLIP.git")

os.system("pip install git+https://github.com/deepmind/dm-haiku")
os.system("pip install ftfy")

os.system("git clone https://github.com/LuminosoInsight/python-ftfy")
 
os.system("pip install cairosvg")
 
os.system("pip install spacy-langdetect")
 
os.system("pip install langid")
 
os.system("pip install tfr_image")
 
os.system("pip install py7zr")
 
import py7zr

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

# Change the current working directory
os.chdir('./python-ftfy')

os.system("python ./setup.py install")
import ftfy
ftfy.fix_text('âœ” No problems')

os.chdir('../')
