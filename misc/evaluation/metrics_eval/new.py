import torch
pt = torch.load('/home/zhoupengfei886/cache/Usada Pekora.pt')
print(list(pt.keys()))
for k,v in pt.items():
    # print(f"{k}:{type(v)}")
    print(f"{k}:{v}")
print(pt['string_to_param']['*'].size())
