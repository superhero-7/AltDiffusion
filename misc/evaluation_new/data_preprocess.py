# %%
import json
from tqdm import tqdm

data = []
path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/crossmodal/captions.jsonl'
with open(path, 'r', encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        data.append(d)
        # print(data)

# %%
language_list = ["tr", "zh", "vi", "ar", "uk", "nl", "ko", "es", "de", "ja", "it", "hi", "pt", "pl", "ru", "th", "fr", "en"]
lan2caption = {}

crossmodal_img_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/crossmodal/images'
# source caption是英文
# target caption是其他语言
for datum in tqdm(data):
    for key in datum.keys():
        if key in language_list:
            if key not in lan2caption:
                lan2caption[key] = []
            if key == 'en':
                lan2caption[key].append(
                    {
                        'source_caption': datum[key]['caption'],
                        'image': crossmodal_img_path + '/' + datum['image/key'] + '.jpg'
                    }
                )
            else:
                lan2caption[key].append(
                    {
                        'target_caption': datum[key]['caption'],
                        'image': crossmodal_img_path + '/' + datum['image/key'] + '.jpg'
                    }
                )

# %%
lan2caption

# %%
import json
with open('./lan2caption.json', 'r') as fn:
    # json.dump(lan2caption, fn)
    lan2caption = json.load(fn)
# %%
for k, v in lan2caption.items():
    print("{} Total length is {}".format(k, len(v)))
    print("Saving {}".format(k))
    
    with open('./json_files/crossmodal_{}.json'.format(k), 'w') as fn:
        json.dump(v, fn)
    
# %%

path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/crossmodal_ar.json'
with open(path, 'r') as fn:
    d = json.load(fn)
d[3599]
# %%
for datum in d:
    if 'source_caption' not in datum:
        print(datum)

# %%
from PIL import Image

img = Image.open(d[3599]['image'])
img

# %%
from glob import glob

path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/'
file_names = glob(path+'*')

for file_name in file_names:
    if 'en' in file_name:
        continue
    with open(file_name, 'r') as fn:
        d = json.load(fn)

    for datum in d:
        datum['target_caption'] = datum['target_caption'][0]
    
    with open(file_name, 'w') as fn:
        json.dump(d, fn)


# %%
import  json
from glob import glob
en_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/crossmodal_en.json'
file_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/*'

filenames = glob(file_path)

d_en = json.load(open(en_path))

img2caption_en = {}

for datum in d_en:
    
    img_name = datum['image'].split('/')[-1]
    img2caption_en[img_name] = datum['source_caption'][0]
# %%
img2caption_en


# %%
for file_name in filenames:
    if 'en' not in file_name:
        d = json.load(open(file_name))
        
    for datum in d:
        img_name = datum['image'].split('/')[-1]
        datum['caption'] = img2caption_en[img_name]
        
    with open(file_name, 'w') as fn:
        json.dump(d, fn)

# %%
import  json
from glob import glob

file_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/json_files/*'

filenames = glob(file_path)
# %%
save_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/generated_imgs'
for filename in filenames:
    key = filename.split('/')[-1]
    key = key.split('.')[0]
    final_dir = save_dir + '/' + key
    data = json.load(open(filename))
    
    for datum in data:
        original_dir = datum['image']
        file_name = original_dir.split("/")[-1]
        datum['generated_image'] = final_dir + '/' + file_name
    
    with open(filename, 'w') as fn:
        json.dump(data, fn)
    
# %%
