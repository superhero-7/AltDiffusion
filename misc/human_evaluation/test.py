# %%
data = []
path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/prompts/prompt_zh.txt'
with open(path, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        data.append(line)
        # print(line)
# %%
data[0]
# %%
s = open('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results_backup/zh/0/altdiffusion/prompt.txt', 'r').read()

s
# %%
import json

chose_idx = json.load(open('/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/choose_3.json', 'r'))

# %%
chose_idx
# %%
import os
from glob import glob
from tqdm import tqdm

for k,v in tqdm(chose_idx.items()):
    img_names = glob(k+'/*.jpg')
    for img_name in img_names:
        idx = img_name.split('/')[-1].split('.')[0]
        idx = int(idx)
        if idx not in v:
            os.remove(img_name)
# %%
type(chose_idx['/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results_backup/vi/8/altdiffusion/'][0])
# %%
from glob import glob
import json

root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/wit_json_files/*'

file_names = glob(root)

for file_name in file_names:
    data = json.load(open(file_name, 'r'))
    
    for datum in data:
        datum['caption'] = datum['source_caption']
        
    with open(file_name, 'w') as fn:
        json.dump(data, fn)
# %%
