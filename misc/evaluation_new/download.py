# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler


# # model_base = '/share/project/yfl/database/ckpt/yfl/stable_diffusion_v2-1'
# model_base = 'stabilityai/stable-diffusion-2-base'
# pipe = StableDiffusionPipeline.from_pretrained(model_base)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.to("cuda:1")
# # # %%

# # prompts = ['cat', 'dog', 'cow']
# # images = pipe(prompts, 
# #     height=512,
# #     width=512, 
# #     num_inference_steps=50,
# #     guidance_scale=8.5)
# # # %%
# # for image in images[0]:
# #     image.show()

# import torch
# from PIL import Image
# from flagai.auto_model.auto_loader import AutoLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loader = AutoLoader(
#     task_name="txt_img_matching",
#     model_name="AltCLIP-XLMR-L-m18",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
#     model_dir="./checkpoints"
# )

# model = loader.get_model()
# tokenizer = loader.get_tokenizer()
# transform = loader.get_transform()

# model.eval()
# model.to(device)
# tokenizer = loader.get_tokenizer()

# def inference():
#     image = Image.open("./dog.jpeg")
#     image = transform(image)
#     image = torch.tensor(image["pixel_values"]).to(device)
#     tokenizer_out = tokenizer(["a rat", "a dog", "a cat"], 
#                                 padding=True,
#                                 truncation=True,
#                                 max_length=77,
#                                 return_tensors='pt')

#     text = tokenizer_out["input_ids"].to(device)
#     attention_mask = tokenizer_out["attention_mask"].to(device)
#     with torch.no_grad():
#         image_features = model.get_image_features(image)
#         text_features = model.get_text_features(text, attention_mask=attention_mask)
#         text_probs = (image_features @ text_features.T).softmax(dim=-1)

#     print(text_probs.cpu().numpy()[0].tolist())

# if __name__=="__main__":
#     inference()

from japanese_stable_diffusion import JapaneseStableDiffusionPipeline
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

model_id = "rinna/japanese-stable-diffusion"
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe = JapaneseStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler).to("cuda")
