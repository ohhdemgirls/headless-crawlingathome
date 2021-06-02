import crawlingathome as cah

client = cah.init(
    url="http://crawlingathome.duckdns.org/",
    nickname= "rvencu"
)

import json, requests, os
from pathlib import Path

def refreshToken(client_id, client_secret, refresh_token):
    params = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    }

    authorization_url = "https://www.googleapis.com/oauth2/v4/token"

    r = requests.post(authorization_url, data=params)

    if r.ok:
        return r.json()['access_token']
    else:
        return None

 
def uploadGdrive(output_filename):
    #output_filename = Path(output_filename).name

    access_t = refreshToken("648172777761-onv1nc5f93nhlhf63flsq6onrmjphpfo.apps.googleusercontent.com","HZ4Zw-_jVJ-3mwicz1NM5W5x", "1//04N2Kysz1LObLCgYIARAAGAQSNwF-L9IrntHNWi2_nEVu2QX5fmlW0Ea0qA-ToBJLSdatDATYxiKcNFI8eZQ_fYN53gjF7b8MGmA")                                                                                                   

    headers = {"Authorization": "Bearer " + access_t} #put ur access token after the word 'Bearer '

    para = {
        "name": output_filename.split("/")[-1], # file name to be uploaded
        "parents": ["1CIgcIR7nX2xNBPB577jwEqbbwxAJR_nt"] # make a folder on drive in which you want to upload files; then open that folder; the last thing in present url will be folder id
    }
    
    files = {
        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
        'file': ('application/zip',open( output_filename , "rb")) # replace                        
    }
    r = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files
    )

while client.jobCount() > 0:

    os.system("mkdir ./save")
    os.system("rm ./save/*")
    
    
    os.system("mkdir ./save/images")
    os.system("rm ./save/images/*")

    os.system("mkdir ./finished")
    os.system("rm ./finished/*")


    # Crawling@Home
    client.newJob()
    client.downloadShard() # Shard is located at './shard.wat'
    
    
    output_folder= "./save/" 
    csv_output_folder = output_folder
    img_output_folder = "./save/images/"  # these image files will be deleted after creating the arrays with CLIP preprocessing
    
    FIRST_SAMPLE_ID_IN_SHARD = client.start_id
    LAST_SAMPLE_ID_IN_SHARD = client.end_id
    N_SAMPLES_IN_SHARD = LAST_SAMPLE_ID_IN_SHARD -FIRST_SAMPLE_ID_IN_SHARD 
    shard_of_chunk = client.shard_piece  # should have values 0 - 1 ; says which 50% of a chunk will be processed
    

    import multiprocessing
    n_processes = min(multiprocessing.cpu_count() * 2, 16)
    
    similarity_threshold = 0.3
    
    import time
    start_time = time.time()
    import os
    from pathlib import Path
    
    import gc
    #import torch.nn as nn
    #cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    from zipfile import ZipFile
    import zipfile
    import zlib
    import pickle
    
    
    content = []
    counter = 0
    with open('shard.wat', 'rb') as infile:
        for line in infile:
            counter += 1
            content.append(line)
    #hard_of_chunk= 0
    if shard_of_chunk ==0: 
        content= content[int(len(content)*0.0):int(len(content)*0.5)]     
    if shard_of_chunk ==1: 
        content= content[int(len(content)*0.5):]    


    def worker(content,index_w, FIRST_SAMPLE_ID_IN_CHUNK, csv_output_folder,img_output_folder, n_processes):
    
        time_out=0.8
        import grequests
        import os
        import time
        import json
        from collections import OrderedDict
        try:
          import cairosvg
        except:
          print("cannot load cairo")
        import pickle
    
        #print("Number of lines in the chunk: " + str(len(content)))
        from pathlib import Path
    
        import spacy
        from spacy_langdetect import LanguageDetector
        import langid
        import time
        from IPython.display import clear_output 
    
        processed_contentlines= 0
        urls=[]
        alttexts=[]
    
        dedupe_urls=[]
        dedupe_alttexts=[]
    
        processed_samples =[]
        #translator = Translator()
    
        CURRENT_SAMPLE_ID=FIRST_SAMPLE_ID_IN_CHUNK
        #alt_text_count = 0
        #working_alt_text_count = 0
    
        #start_time = time.time()
        for line in content:
        #print(line)
          line_str =line.decode("utf-8")
          alt_text_result=0
          img_result=0
    
          img_result =line_str.find("IMG@")
          processed_contentlines += 1
          if img_result>0:
            alt_text_result = line_str.find("alt\":")
            
            if alt_text_result>0:
              #print(line_str)
              data = json.loads(line_str)
              
              linklist= data['Envelope']['Payload-Metadata']['HTTP-Response-Metadata']['HTML-Metadata']['Links']
              #print(data['Envelope']['Payload-Metadata']['HTTP-Response-Metadata']['HTML-Metadata'])
    
              for e in linklist:
                if "alt" in e:
                  
                  if len(e["alt"]) >4:
                    #print(e["alt"])
    
                    try:
                      if e["url"][0] =="h" and not e["url"] in dedupe_urls and not e["alt"] in dedupe_alttexts:
                        
                        urls.append(e["url"])
                        alttexts.append(e["alt"].encode("ascii", "ignore").decode())
                        dedupe_urls.append(e["url"])
                        dedupe_alttexts.append(e["alt"].encode("ascii", "ignore").decode())
                        #print("***")
                        #print("len(urls) "+str(len(urls)))
                        #print("len(alttexts) "+str(len(alttexts)))
                        #print("len(e[alt]) "+str(len(e["alt"])))
                
                    except:
                      continue
    
          if len(urls)>2000:
            
                
            try:
              # Once the last line of content is filtered, send the last requests
              rs = (grequests.get(u, timeout=time_out) for u in urls)
    
              responses = grequests.map(rs)
            except:
              continue
    
            for i in range (len(responses)):
              try:
                
                if responses[i].status_code == 200 and  responses[i].headers['Content-Type'][:3]=="ima" and len(responses[i].content)>5000:
    
                    #print(urls[i])
                    filetype = Path(urls[i]).suffix    #os.path.splitext(e["url"])[1]
                    #print(filetype)
    
                    if len(filetype)<4:
                      continue
    
                    if filetype[:4] ==".jpg" or filetype[:4] ==".JPG" or filetype[:4] ==".png" or filetype[:4] ==".PNG" or filetype[:4] ==".svg" or filetype[:4] ==".SVG" or filetype[:4] ==".gif" or filetype[:4] ==".GIF" or filetype[:4] ==".bmp" or filetype[:4] ==".BMP" or filetype[:4] ==".tif" or filetype[:4] ==".TIF":
                      if len(filetype)>4:
                        filetype = filetype[:4] 
    
    
                    if (filetype[:5] ==".webp" or filetype[:5] ==".WEBP" or filetype[:5] ==".jpeg" or filetype[:5] ==".JPEG")  and len(filetype)>5:
                      filetype = filetype[:5] 
    
                    if filetype[:4] != ".jpg" and filetype[:4] != ".JPG" and filetype[:4] != ".png" and filetype[:4] !=".PNG" and filetype[:4] !=".svg" and filetype[:4] !=".SVG" and filetype[:4] !=".gif" and filetype[:4] != ".GIF" and filetype[:4] !=".bmp" and filetype[:4] !=".BMP" and filetype[:4] !=".tif" and filetype[:4] !=".TIF"and filetype[:5] !=".webp" and filetype[:5] !=".WEBP" and filetype[:5] !=".jpeg" and filetype[:5] !=".JPEG":
                      continue
                        
                    if filetype == ".svg":
                      #print("SVG found")
                      try:
                        output_filename= img_output_folder + str(CURRENT_SAMPLE_ID)+filetype
                        cairosvg.svg2png( url=urls[i], write_to=output_filename, output_height=600)
                        #print("SVG converted")
                      except:
                        continue
                    else:
                      try:
                          img_data = responses[i].content
                          with open(img_output_folder + str(CURRENT_SAMPLE_ID)+filetype, 'wb') as handler:
                              handler.write(img_data)
                          #print("Saved: "+img_output_folder + str(CURRENT_SAMPLE_ID)+filetype)
                      except:
                        continue
    
                    processed_samples.append([CURRENT_SAMPLE_ID, urls[i], alttexts[i]])
                    
                    CURRENT_SAMPLE_ID +=1
                    
                  
              except:
                continue
    
            urls=[]
            alttexts=[]
    
    
            
    
            #print("Worker "+str(index_w)+"--- %s seconds ---" % (time.time() - start_time))
            
            #print("processed_samples-len: "+str(len(processed_samples)))
            #print("processed_samples[0]-len: "+str(len(processed_samples[0])))
    
            #print("Currently processed content lines: "+str(processed_contentlines))
    
            #clear_output()
        #print("last filtering")
        #print(len(urls))
        try:
          # Once the last line of content is filtered, send the last requests
          rs = (grequests.get(u, timeout=time_out) for u in urls)
    
          responses = grequests.map(rs)
        except:
          responses=[]
        
        #print("len(responses): " + str(len(responses)))
        #print(responses)
        for i in range (len(responses)):
          if responses[i]==None:
            continue
          #print(i)
          #print(responses[i].status_code)
          #print(responses[i].headers['Content-Type'][:3])
          #print(len(responses[i].content))
          try:
            
            if responses[i].status_code == 200 and  responses[i].headers['Content-Type'][:3]=="ima" and len(responses[i].content)>5000:
                #print(urls[i])
                filetype = Path(urls[i]).suffix 
                if len(filetype)<4:
                  continue
    
                if filetype[:4] ==".jpg" or filetype[:4] ==".JPG" or filetype[:4] ==".png" or filetype[:4] ==".PNG" or filetype[:4] ==".svg" or filetype[:4] ==".SVG" or filetype[:4] ==".gif" or filetype[:4] ==".GIF" or filetype[:4] ==".bmp" or filetype[:4] ==".BMP" or filetype[:4] ==".tif" or filetype[:4] ==".TIF":
                  if len(filetype)>4:
                    filetype = filetype[:4] 
    
                if (filetype[:5] ==".webp" or filetype[:5] ==".WEBP" or filetype[:5] ==".jpeg" or filetype[:5] ==".JPEG")  and len(filetype)>5:
                  filetype = filetype[:5] 
    
                if filetype[:4] != ".jpg" and filetype[:4] != ".JPG" and filetype[:4] != ".png" and filetype[:4] !=".PNG" and filetype[:4] !=".svg" and filetype[:4] !=".SVG" and filetype[:4] !=".gif" and filetype[:4] != ".GIF" and filetype[:4] !=".bmp" and filetype[:4] !=".BMP" and filetype[:4] !=".tif" and filetype[:4] !=".TIF"and filetype[:5] !=".webp" and filetype[:5] !=".WEBP" and filetype[:5] !=".jpeg" and filetype[:5] !=".JPEG":
                  continue
    
    
                if filetype == ".svg":
                  #print("SVG found")
                  try:
                    output_filename= img_output_folder + str(CURRENT_SAMPLE_ID)+filetype
                    cairosvg.svg2png( url=urls[i], write_to=output_filename, output_height=600)
                    #print("Saved: "+img_output_folder + str(CURRENT_SAMPLE_ID)+filetype)
                  except:
                    continue
                else:
                  try:
    
                    img_data = responses[i].content
                    with open(img_output_folder + str(CURRENT_SAMPLE_ID)+filetype, 'wb') as handler:
                        handler.write(img_data)
                    #print(responses[i].headers)
    
                    
                    #print("Saved: "+img_output_folder + str(CURRENT_SAMPLE_ID)+filetype)
    
                  except:
                    continue
    
                processed_samples.append([CURRENT_SAMPLE_ID, urls[i], alttexts[i] ])
                CURRENT_SAMPLE_ID +=1
                
        
                
    
          except:
            continue
    
    
        import pandas as pd
        sample_ids= []
        sample_urls=[]
        sample_texts=[]

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        for e in processed_samples:

          text= e[2]
          doc = nlp(text[:76])
          # document level language detection. Think of it like average language of the document!
          lang_detection = doc._.language
          
          lang_detection2 = langid.classify(text[:76])
          if lang_detection["language"] == "en" or lang_detection2[0]=="en":

            sample_ids.append(e[0])
            sample_urls.append(e[1])
            sample_texts.append(e[2])

        df = pd.DataFrame(list(zip(sample_ids,sample_urls,sample_texts)),   columns =['SAMPLE_ID', 'URL', 'TEXT'])

        output_filename= csv_output_folder +"FIRST_SAMPLE_ID_"+str(FIRST_SAMPLE_ID_IN_CHUNK) + "__LAST_SAMPLE_ID_"+ str(CURRENT_SAMPLE_ID-1)+"_"+str(n_processes)+".csv"
      
        df.to_csv(output_filename ,sep = '|',header=True, mode='w', index=False)
    
    
    

        
    client.log("Downloading Images")
    #start_time = time.time()
    jobs = []
    for i in range(n_processes):
        print("Starting to scraping part "+str(i)+ " of "+str(n_processes))
        content_chunk =content [i * int(len(content)/n_processes) : (i+1) * int(len(content)/n_processes)] #i*1000000
        FIRST_SAMPLE_ID_IN_CHUNK = FIRST_SAMPLE_ID_IN_SHARD+ i *int(N_SAMPLES_IN_SHARD/n_processes)
        p = multiprocessing.Process(target=worker, args=[content_chunk,i,FIRST_SAMPLE_ID_IN_CHUNK,csv_output_folder, img_output_folder, n_processes])
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    
    import gc
    for p in jobs:
        del p
    gc.collect()
        
    

    ###### 

    def imgfiles_to_embeddings(list_of_files, batch_size, model, preprocess):
      if batch_size<2:
        print("Minimal batch_size is 2 ")
        return []

      import numpy as np
      from PIL import Image

      import time

      import os
      #print(os.getcwd())
      #os.chdir('./dm-haiku')
      #import haiku as hk
      #import torch
      #import clip
      #from PIL import Image




      #device = "cuda" if torch.cuda.is_available() else "cpu"
      #model, preprocess = clip.load("ViT-B/32", device=device)

      #image = preprocess(Image.open("/content/V-300.png")).unsqueeze(0).to(device)
      #text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

      counter_samples =0

      list_of_arrays_to_concat = []
      list_of_tokenized_text_arrays =[]
      list_of_arrays_to_concat = []
      list_of_image_arrays =[]
      list_of_tokenized_text_arrays =[]
      img_embeddings= []



      #devices =  jax.local_devices()

      #print(f"jax devices: {devices}")

      #jax_params = jax.device_put_replicated(jax_params, devices)
      #image_fn = jax.pmap(image_fn)

      for img_path in list_of_files:

        try:
          new_image_array = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except:
          new_image_array = preprocess(Image.new("RGB", (300, 300), (255, 255, 255))).unsqueeze(0).to(device)

        #new_image_array = np.expand_dims(jax_preprocess(Image.open(img_path)), (0, 1))
        #print("new_image_array = np.expand_dims(jax_preprocess(Image.open(img_path)), (0, 1))")
        #print(new_image_array.shape)  torch.cat((image,image,image,image,image,image,image,image,image,image), 0)
        #except:
        #  list_of_samples_to_drop.append(sample_id)
        #  print("dropped sample: "+str(sample_id))
        #  continue


        if counter_samples%batch_size ==0:
          image_array =new_image_array
          #tokenized_text_np_array = tokenized_text_np_array_new_sample
          counter_samples +=1
          continue
        else:
          image_array =  torch.cat((image_array,new_image_array), 0)
          counter_samples +=1
          #print(image_array.shape)


        
        if counter_samples%batch_size ==0:
            with torch.no_grad():
              image_features = model.encode_image(image_array)
              

              for i in range (image_features.shape[0]):
                  img_embeddings.append(torch.reshape(image_features[i], (1, 512)))
                  #img_embeddings.append(image_features[i])
                  #print(torch.reshape(image_features[i], (1, 512)) .shape)
      with torch.no_grad():
          image_features = model.encode_image(image_array)
          for i in range (image_features.shape[0]):
            img_embeddings.append(torch.reshape(image_features[i], (1, 512)))
            #img_embeddings.append(image_features[i])

      #print(len(img_embeddings))
      #print(img_embeddings[0].shape)
      return img_embeddings




    import sys
    
    
    import glob
    import pandas as pd
    
    #import psutil
    
    # Change the current working directory
    #os.chdir('/content/dm-haiku')
    #print(os.getcwd())
    
    from PIL import Image
    import time
    import numpy as np
    
    batch_size = 512 #2048
    
    
    client.log("Saving CSV")
    
    
    counter_samples=0
    
    
    #list_of_text_arrays_to_concat = []
    
    
    #list_of_tokenized_text_arrays =[]
    sample_ids = []
    
    #text_array_dict={}
    
    c=0
    csv_files = glob.glob(output_folder + "*.csv")
    #print(csv_files)
    rows=0
    for csv_file in csv_files:
    
        df = pd.read_csv(csv_file, sep = '|',lineterminator='\n')
            
        rows += len(df)
        if c ==0:
            #print(c)
            df.to_csv(output_folder + 'FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+"_"+ str(shard_of_chunk)+'.csv',sep = '|',header=True, mode='a', index=False)
        else:
            #print(c)
            df.to_csv(output_folder + 'FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+"_"+ str(shard_of_chunk)+'.csv',sep = '|',header=False, mode='a', index=False)
    
        #print(rows)
        c +=1
        os.remove(csv_file)
    
    
    
    
    #from sklearn.metrics.pairwise import cosine_similarity
    #from vectorhub.bi_encoders.text_image.torch import Clip2Vec
    
    #model = Clip2Vec()

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch.nn as nn
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    csv_file=output_folder + 'FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+"_"+ str(shard_of_chunk)+'.csv'
    
    
    df = pd.read_csv(csv_file, sep = '|',lineterminator='\n')
    print ("len(df) before filtering with clip"+str(len(df)))
    
    img_files = glob.glob(img_output_folder + "*.*")
    img_files_ids ={}
    img_ids_by_filepath={} 
    for img_path in img_files:
        path = Path(img_path)
        path.name
        img_files_ids[path.stem]= img_path
        img_ids_by_filepath[img_path] = path.stem
    
    #print(img_files_ids)

    import torch
    import clip
    from PIL import Image




    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    batch_size =512
    img_emb_list= imgfiles_to_embeddings(img_files, batch_size, model, preprocess)

    #print("len(img_files)")
    #print(len(img_files))


    #print("len(img_emb_list)")
    #print(len(img_emb_list))

    image_embedding_dict = {}

    c= 0
    for path in img_files:
        img_sample_id = img_ids_by_filepath[path]
        image_embedding_dict[img_sample_id] = img_emb_list[c]

        c +=1

    #print("len(image_embedding_dict)")
    #print(len(image_embedding_dict))


    untokenized_texts=[]

    tokenized_texts=[]
    sample_ids_tokenized_texts=[]

    text_embedding_list = []
    for row_index, row in df.iterrows():
        untokenized_texts.append (str( df.at[row_index,'TEXT']) [:75])
        sample_ids_tokenized_texts.append (df.at[row_index,'SAMPLE_ID'])
        if row_index% 128 ==0 and row_index >0:


            tokenized_texts = clip.tokenize(untokenized_texts).to(device)
            with torch.no_grad():
              text_embeddings = model.encode_text(tokenized_texts)
            for i in range(text_embeddings.shape[0]):
              text_embedding_list.append(text_embeddings[i])

            untokenized_texts=[]

    if len(untokenized_texts)>0:      
        tokenized_texts = clip.tokenize(untokenized_texts).to(device)

        with torch.no_grad():
          text_embeddings = model.encode_text(tokenized_texts)
        for i in range(text_embeddings.shape[0]):
          text_embedding_list.append(text_embeddings[i])
        untokenized_texts=[]



    
    #texts_by_sample_id_dict[df.at[row_index,'SAMPLE_ID'] ] 
    #### NSFW detector categories text embeddings
    
    #0-18 /first 19 are not NSFW
    nsfw_text_categories = ["neutral","selfie", "illustration, drawng", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]

    nsfw_text_tokenized = clip.tokenize(nsfw_text_categories).to(device)
    nsfw_text_features =[]
    with torch.no_grad():
      nsfw_text_embed = model.encode_text(nsfw_text_tokenized)

    for i in range(nsfw_text_embed.shape[0]):
        nsfw_text_features.append(nsfw_text_embed[i])

    #nsfw_text_features = np.array_split(nsfw_text_embed, len(nsfw_text_categories))

    
    
    
    
    listofzeros = ["-"] * len(df)
    
    df["NSFW"]=listofzeros
    
    
    
    #first 4 are underaged, 0-3
    underaged_categories = ["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]
    

    underaged_text_tokenized = clip.tokenize(underaged_categories).to(device)
    underaged_text_features =[]
    with torch.no_grad():
      underaged_text_embed = model.encode_text(underaged_text_tokenized)

    for i in range(underaged_text_embed.shape[0]):
        underaged_text_features.append(underaged_text_embed[i])

    
    


    #0-20 /first 21 are not animals
    animal_categories = ["lifelss object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]
    
    animal_text_tokenized = clip.tokenize(animal_categories).to(device)
    animal_text_features =[]
    with torch.no_grad():
      animal_text_embed = model.encode_text(animal_text_tokenized)

    for i in range(animal_text_embed.shape[0]):
        animal_text_features.append(animal_text_embed[i])


    #print(len(animal_categories))  
    #print(len(animal_text_features))  
    ######### 

    
    # given an iterable of pairs return the key corresponding to the greatest value
    def argmax(pairs):
        return max(pairs, key=lambda x: x[1])[0]
    
    # given an iterable of values return the index of the greatest value
    def argmax_index(values):
        return argmax(enumerate(values))
    
    
    listofzeros = [0.0] * len(df)
    
    df["similarity"]=listofzeros
    
    #image_embedding_dict= {}
    #print ("len(df)"+str(len(df)))
    
    img_dict_counter= 0
    #print ("len(df) before 1st for row_index, row in df.iterrows():"+str(len(df)))


    client.log("Dropping NSFW Keywords")
    
    
    for row_index2, row2 in df.iterrows():
        if str(df.at[row_index2,'TEXT']).lower().find("sex") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("nude") !=-1  or  str(df.at[row_index2,'TEXT']).lower().find("sexy") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("fuck") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("orgasm") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("porn") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("lesbian") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("lust") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("pussy") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("bdsm") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("titts") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("horny") !=-1   or str(df.at[row_index2,'TEXT']).lower().find("nacked") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("boops") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("erotic") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("lingerie") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("penis") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("dick") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("cock") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("dig") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("clit") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("nipple") !=-1  or str(df.at[row_index2,'TEXT']).lower().find("gay") !=-1  :
            
            if str(df.at[row_index2,'TEXT']).lower().find("teen") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("kid") !=-1  or  str(df.at[row_index2,'TEXT']).lower().find("child") !=-1 or str(df.at[row_index2,'TEXT']).lower().find("baby") !=-1 :           
            
                #print(###########NSFW KEYWORD DROP##############)
            
                #print (df.at[row_index2,'TRANSLATION']))
                df = df.drop(row_index2)
                continue
    
    
    similarity_counter= 0
    for row_index, row in df.iterrows():
        try:
    
        #if similarity_counter>500 and similarity_counter%10==0:
            #!nvidia-smi
    
            if row_index % 100 ==0:
                print("row_index: "+ str(row_index))
                client.log(f"Removing NFSW: {row_index} / ?")

            sample_id = df.at[row_index,'SAMPLE_ID']
            index_of_row_in_list= sample_ids_tokenized_texts.index(sample_id)
            #print("index_of_row_in_list")
            #print(index_of_row_in_list)
            if index_of_row_in_list==-1:
                df = df.drop(row_index)
                continue

            current_text_embedding = text_embedding_list[index_of_row_in_list]
            current_image_embedding = image_embedding_dict[str(sample_id)]
            #print("current_image_embedding")
            #print(current_image_embedding.shape)
            #print("current_text_embedding")
            #print(current_text_embedding.shape)
            similarity= float (cosine_similarity(torch.reshape(current_text_embedding, (1, 512)) , current_image_embedding )) 
            print(df.at[row_index,'TEXT'])
            print(df.at[row_index,'URL'])
            print("similarity:")

            print(similarity)
            if similarity > similarity_threshold:
                df.at[row_index,'similarity'] = similarity
                similarity_counter +=1



                #0-18 /first 19 are not NSFW
                nsfw_text_categories = ["neutral","selfie", "illustration, drawng", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]
                #nsfw_text_features = model.encode_text(nsfw_text_categories)
                similarities=[]
        
                for i in range(len(nsfw_text_features)):
                    similarity= float (cosine_similarity(torch.reshape(nsfw_text_features[i], (1, 512)) , current_image_embedding )) 
                    similarities.append( similarity )
        
                print(similarities)
        
                argmax1= argmax_index(similarities)
                most_likely= nsfw_text_categories[argmax1]
                print ("most_likely")
                print (most_likely)
        
        
                nsfw_text_categories.pop(argmax_index(similarities))
                similarities.pop(argmax_index(similarities))
                argmax2= argmax_index(similarities)
                second_likely = nsfw_text_categories[argmax_index(similarities)]
        
                if argmax1 <19 and argmax2<19:
                    df.at[row_index,'NSFW'] = "UNLIKELY"
                elif argmax1 <19 and argmax2>=19:
                    df.at[row_index,'NSFW'] = "UNSURE"
                elif argmax2 <19 and argmax1>=19:
                    df.at[row_index,'NSFW'] = "UNSURE"
                elif argmax1 >=19 and argmax2>=19:
                    df.at[row_index,'NSFW'] = "NSFW"
        

        
                ####underaged check 
                if df.at[row_index,'NSFW'] != "UNLIKELY":
        
                    #keyword check
                    if str(df.at[row_index,'TEXT']).lower().find("teen") !=-1 or str(df.at[row_index,'TEXT']).lower().find("kid") !=-1  or  str(df.at[row_index,'TEXT']).lower().find("child") !=-1 or str(df.at[row_index,'TEXT']).lower().find("baby") !=-1 :
                        df = df.drop(row_index)
                        print(###########NSFW KEYWORD DROP##############)
                        print (df.at[row_index,'TEXT']))
                        continue
                    
                    #first 4 are underaged, 0-3
                    underaged_categories = ["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age", "drawing, logo, clip art", "illustration, cartoon", "captcha, screen", "food, eating, meal, drink", "car"]
        
                    similarities=[]
        
                    for i in range(len(underaged_text_features)):
                        #similarities.append( cosine_similarity([underaged_text_features[i][0]], [current_image_embedding[0][0]]) )
            
                        similarity= float (cosine_similarity(torch.reshape(underaged_text_features[i], (1, 512)) , current_image_embedding )) 
                        similarities.append( similarity )        
        
                    argmax1= argmax_index(similarities)
                    print("argmax1")
                    print(argmax1)
                    most_likely= underaged_categories[argmax1]
                    
                    print ("most_likely")

                    print (most_likely)
        
                    underaged_categories.pop(argmax_index(similarities))
                    similarities.pop(argmax_index(similarities))
                    argmax2= argmax_index(similarities)
                    print("argmax2")
                    print(argmax2)
                    second_likely = underaged_categories[argmax_index(similarities)]
                    print(second_likely)
                    if argmax1 <4 or argmax2 <4:
                        #print( df.at[row_index,'URL'] )
                        del image_embedding_dict[str(sample_id)]
                        df = df.drop(row_index)

                        print("dropped cause NSFW and eventually underaged")
                        
                        continue
        
        
                ####animal check 
                if df.at[row_index,'NSFW'] != "UNLIKELY":
                    
                    #0-20 /first 21 are not animals
                    animal_categories = ["lifelss object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]
        
                    similarities=[]
        

                    for i in range(len(animal_text_features)):
                        #similarities.append( cosine_similarity([animal_text_features[i][0]], [current_image_embedding[0][0]]) )
                        similarity= float (cosine_similarity(torch.reshape(animal_text_features[i], (1, 512)) , current_image_embedding )) 
                        similarities.append( similarity )       
                    print ("most_likely")

                    print (most_likely)
        
                    argmax1= argmax_index(similarities)
                    most_likely= animal_categories[argmax1]
        
        
                    #print(second_likely)
                    if argmax1 >20:

                        del image_embedding_dict[str(sample_id)]

                        df = df.drop(row_index)
                        print("dropped cause NSFW and eventually animal")
                        
                        continue

            else:
                del image_embedding_dict[str(sample_id)]
                df = df.drop(row_index)
                continue

        except Exception as e:
            #print("dropped sample: "+str(df.at[row_index,'SAMPLE_ID']))
            print(e)
            print( "embedding error")
            
            try:
                df = df.drop(row_index)
            except:
                print("WEIRD ERROR")
            continue
    
            
    df.reset_index(drop=True, inplace=True)

    client.log("Dumping Embeddings")
    for key in image_embedding_dict:
      image_embedding_dict[key] = image_embedding_dict[key].cpu().detach().numpy()
    
    # save the last img_array_dict
    with open(output_folder + "image_embedding_dict"+ '-FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+"_"+ str(shard_of_chunk)+".pkl","wb") as f:
        pickle.dump(image_embedding_dict, f)
    
    '''
    with ZipFile(output_folder + "img_array_dict_"+'FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD) +'.zip','w',zipfile.ZIP_DEFLATED) as zip:
            zip.write(output_folder + "image_embedding_dict"+ '_FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+".pkl")
    os.remove(output_folder + "image_embedding_dict"+ '_FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+".pkl")
    '''
    
    
    
    ### converting core data to TFrecords
    
    
    #!pip install tfr_image
    
    
    from tfr_image import TFRimage
    
    from tfr_image.utils import (
        get_filenames_and_classes,
        write_label_file,
        bytes_feature,
        int64_feature,
    )
    
    
    def _get_dataset_filename(
            dataset_dir, split_name, shard_id, tfrecord_filename, num_chards
        ):
            output_filename = "%s_%s_%05d-of-%05d.tfrecord" % (
                tfrecord_filename,
                split_name,
                shard_id,
                num_chards,
            )
            return os.path.join(dataset_dir, output_filename)
    
    
    
    def _image_to_tfexample( sampleID, image_data, image_format, height, width, caption):
    
    
            return tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "sampleID": bytes_feature(sampleID),
                        "image": bytes_feature(image_data),
                        "format": bytes_feature(image_format),
                        "label": bytes_feature(caption),#int64_feature(class_id),
                        "height": int64_feature(height),
                        "width": int64_feature(width),
                    }
    
    
                )
            )
    
    
    def _convert_dataset(
            
            split_name,
            filenames,
            sampleIDs,
            captions,
            dataset_dir,
            tfrecord_filename,
            num_chards,
            df
        ):
            """Converts the given filenames to a TFRecord dataset.
            Args:
            split_name: The name of the dataset, either 'train' or 'validation'.
            filenames: A list of absolute paths to png or jpg images.
            class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
            dataset_dir: The directory where the converted datasets are stored.
            """
            #assert split_name in ["train", "validation"]
    
            num_per_shard = int(math.ceil(len(filenames) / float(num_chards)))
    
            for shard_id in range(num_chards):
                #if shard_id % 100 == 0:
                #    client.log(f"Converting to TFRs: {shard_id} / {len(num_chards)}")
                
                output_filename = _get_dataset_filename(
                    dataset_dir,
                    split_name,
                    shard_id,
                    tfrecord_filename=tfrecord_filename,
                    num_chards=num_chards,
                )
                heights=[]
                widths=[]
                with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(len(filenames)):
                        sys.stdout.write(
                            "\r>> Converting image %d/%d shard %d"
                            % (i + 1, len(filenames), shard_id)
                        )
                        sys.stdout.flush()
    
                        # Read the filename:
                        try:
                            image_data = tf.io.gfile.GFile(filenames[i], "rb").read()
                            with tf.io.gfile.GFile(filenames[i], "rb") as f:
                                image = Image.open(f)
                        except:
                            df = df.drop(i)
                            continue
                        height, width = image.size
                        heights.append(height)
                        widths.append(width)
                        #class_name = os.path.basename(os.path.dirname(filenames[i]))
                        #class_id = class_names_to_ids[class_name]
                        
                        try:
                            caption = captions[i].encode('utf_8')
                            sampleID = sampleIDs[i].encode('utf_8')
                        except:
                            caption = captions[i]
                            sampleID = sampleIDs[i]
    
                        example = _image_to_tfexample( sampleID,
                            image_data, b"jpg", height, width, caption # class_id
                        )
                        tfrecord_writer.write(example.SerializeToString())
            df.reset_index(drop=True, inplace=True)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return widths,heights, df
    
    
    import random
    import math
    #import os
    import sys
    from PIL import Image
    import tensorflow as tf
    
    
    sampleIDs= []
    image_filenames=[]
    translations=[]
    for row_index, row in df.iterrows():
        sampleIDs.append(str(df.at[row_index,'SAMPLE_ID']))
        translations.append(str(df.at[row_index,'TEXT']))
        image_filenames.append(img_files_ids[str(df.at[row_index,'SAMPLE_ID'])])
    
    listofzeros = ["-"] * len(df)
    
    df["WIDTH"]=listofzeros
    df["HEIGHT"]=listofzeros
    
    
    widths,heights, df = _convert_dataset(split_name = "", sampleIDs = sampleIDs, filenames=image_filenames, captions=translations, dataset_dir="./save/", tfrecord_filename="crawling_at_home_"+ 'FIRST_SAMPLE_ID_IN_SHARD_'+str(FIRST_SAMPLE_ID_IN_SHARD)+"_LAST_SAMPLE_ID_IN_SHARD_"+str(LAST_SAMPLE_ID_IN_SHARD)+"_"+str(shard_of_chunk), num_chards=1, df=df)
    #print("len(heights)" + str(len(heights)))
    #print("len(df)" + str(len(df)))
    
    for row_index, row in df.iterrows():
        df.at[row_index,'WIDTH'] = widths[row_index]
        df.at[row_index,'HEIGHT'] =heights[row_index]
    
    client.log("Saving TFRs")
    
    df.to_csv(csv_file ,sep = '|',header=True, mode='w', index=False)
    #len (df)
    
    for key in img_files_ids:
        os.remove(img_files_ids[key])
    
    '''
    #remove the remaining img files of the samples that had beed dropped  
    import shutil
    shutil.rmtree(img_output_folder)
    ''' 

    # Now we need to upload for Crawling@Home

    from pathlib import Path

    saves = Path("./save")

    client.log("Uploading CSV")
    uploadGdrive(f"./save/FIRST_SAMPLE_ID_IN_SHARD_{FIRST_SAMPLE_ID_IN_SHARD}_LAST_SAMPLE_ID_IN_SHARD_{LAST_SAMPLE_ID_IN_SHARD}_"+str(shard_of_chunk)+".csv")

    tfrecords = [*saves.glob("*.tfrecord")]
    for i, f in enumerate(tfrecords):
        client.log(f"Uploading TFRs: {i + 1} / {len(tfrecords)}")
        uploadGdrive(str(f))
    
    client.log("Uploading Image Embeddings")
    uploadGdrive(f"./save/image_embedding_dict-FIRST_SAMPLE_ID_IN_SHARD_{FIRST_SAMPLE_ID_IN_SHARD}_LAST_SAMPLE_ID_IN_SHARD_{LAST_SAMPLE_ID_IN_SHARD}_"+str(shard_of_chunk)+".pkl")

    client._markjobasdone(len(df))

    print("Complete time --- %s seconds ---" % (time.time() - start_time))



#client.bye()
