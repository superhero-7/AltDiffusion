import gradio as gr
from PIL import Image
from glob import glob
import random

idx=-1

def main():
    # pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    # example_image = Image.open("imgs/example.jpg").convert("RGB")

    def load_data():
        root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/generated_results_backup'
        
        alt_dict={}
        sd_dict={}
        
        root_lgs = glob(root+'/*')
        # root_lgs = [root+ '/' + 'zh']
        for root_lg in root_lgs:
            lg = root_lg.rsplit('/', 1)[1]
            
            all_prompt_path = glob(root_lg+'/*')
            alt_img_paths = []
            sd_img_paths = []
            for single_prompt_path in all_prompt_path:
                prompt_en = open(single_prompt_path+'/sd2.1/prompt.txt').read()
                prompt_zh = open(single_prompt_path+'/prompt.txt').read()
                prompt_lg = open(single_prompt_path+'/altdiffusion/prompt.txt').read()
                
                alt_img_path_single = single_prompt_path + '/' + 'altdiffusion/*.jpg'
                sd_img_path_single = single_prompt_path + '/' + 'sd2.1/*.jpg'
                
                alt_img_list = glob(alt_img_path_single)
                sd_img_list = glob(sd_img_path_single)
                
                for alt_img_path, sd_img_path in zip(alt_img_list, sd_img_list):
                    alt_img_paths.append(
                        {
                            'image':alt_img_path,
                            'prompt':prompt_lg,
                            'prompt_zh':prompt_zh
                        }
                    )
                    
                    sd_img_paths.append(
                        {
                            'image':sd_img_path,
                            'prompt':prompt_en,
                            'prompt_zh':prompt_zh
                        }
                    )
            alt_dict[lg] = alt_img_paths
            sd_dict[lg] = sd_img_paths
        
        return alt_dict, sd_dict
            
    print("Loading Data...")
    alt_dict, sd_dict = load_data()
    print("Finised !!! ^_^")
    
    
    def next_img(lg, model):
        
        global idx
        idx += 1
                
        if idx<0:
            idx = 0
            
        large_len = len(alt_dict[lg])
        if idx>=large_len:
            idx = large_len-1
        
        alt_itum = alt_dict[lg][idx]
        sd_itum = sd_dict[lg][idx]
        
        if model=='Altdiffusion':
            alt_image_path = alt_itum['image']
            img = Image.open(alt_image_path)
            img_path = alt_image_path
        
        if model=='Stable Diffusion':
            sd_image_path = sd_itum['image']
            img = Image.open(sd_image_path)
            img_path = sd_image_path
        
        en_prompt = sd_itum['prompt']
        zh_prompt = alt_itum['prompt_zh']
        lg_prompt = alt_itum['prompt']
        
        return [img, en_prompt, zh_prompt, lg_prompt, img_path]

    def pre_img(lg, model):
        
        global idx
        idx -= 1
        
        if idx<0:
            idx = 0
            
        large_len = len(alt_dict[lg])
        if idx>=large_len:
            idx = large_len-1
        
        alt_itum = alt_dict[lg][idx]
        sd_itum = sd_dict[lg][idx]
        
        if model=='Altdiffusion':
            alt_image_path = alt_itum['image']
            img = Image.open(alt_image_path)
            img_path = alt_image_path
        
        if model=='Stable Diffusion':
            sd_image_path = sd_itum['image']
            img = Image.open(sd_image_path)
            img_path = sd_image_path
        
        en_prompt = sd_itum['prompt']
        zh_prompt = alt_itum['prompt_zh']
        lg_prompt = alt_itum['prompt']
        
        return [img, en_prompt, zh_prompt, lg_prompt, img_path]

    # result_dict = {
    #     'alt':0,
    #     'sd':0,
    # }
    
    def submit(ans, lg, model, img_path):
            
        root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/result'
        save_path = root + '/' + lg + '.txt'
        with open(save_path, "a") as file:
            file.write(img_path)
            file.write('\t')
            file.write(str(ans) + '\n')
        
        return next_img(lg, model)
        
            
    # def reset():
    #     result_dict['alt'] = 0
    #     result_dict['sd'] = 0

    # def save(lg):
    #     root = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/result'
    #     save_path = root + '/' + lg + '.txt'
    #     with open(save_path, "a") as file:
    #         # 将字典转换为字符串并写入文件
    #         for key, value in result_dict.items():
    #             file.write(f"{key}: {value}\n")
        
    with gr.Blocks() as demo:

        with gr.Row():
            image = gr.Image(label="Image", type="pil", interactive=False)
            # sd_image = gr.Image(label="SD Image", type="pil", interactive=False)
            image.style(height=512, width=512)
            # sd_image.style(height=512, width=512)

        with gr.Row():
            with gr.Column():
                en_prompt = gr.Textbox(label="En prompt")
            with gr.Column():
                zh_prompt = gr.Textbox(label="Zh prompt")
            with gr.Column():
                lg_prompt = gr.Textbox(label="Lg prompt")

        with gr.Row():
            lg = gr.Dropdown(["ar", "de", "es", "fr", "hi", "it", "ja", "ko", "nl", "pl", "pt", 
                              "ru", "th", "tr", "uk", "vi", "zh"], label='language', info='Please choose a language to evaluation.')
            model = gr.Dropdown(["Altdiffusion", "Stable Diffusion"], label='model', info='Please choose model language to evaluation.')
            with gr.Column(scale=1, min_width=100):
                next_button = gr.Button("Next")
            with gr.Column(scale=1, min_width=100):
                pre_button = gr.Button("Pre")

        with gr.Row():
            with gr.Column():
                result = gr.Radio([1, 2, 3, 4, 5], label="Rate", info="Please give a rate!")
            with gr.Column():
                submit_button = gr.Button("Submit")
            # with gr.Column():
            #     save_button = gr.Button("Save")
            # with gr.Column():
            #     reset_button = gr.Button("Reset")
        
        with gr.Row(visible=False):
            Img_path = gr.Textbox(label="img path")
            

        next_button.click(
            fn=next_img,
            inputs=[
                lg, model
            ],
            outputs=[image, en_prompt, zh_prompt, lg_prompt, Img_path],
        )
        pre_button.click(
            fn=pre_img,
            inputs=[
                lg, model
            ],
            outputs=[image, en_prompt, zh_prompt, lg_prompt, Img_path],
        )
        submit_button.click(
            fn=submit,
            inputs=[result, lg, model, Img_path],
            outputs=[image, en_prompt, zh_prompt, lg_prompt, Img_path],
        )
        # reset_button.click(
        #     fn=reset,
        # )
        # save_button.click(
        #     fn=save,
        #     inputs=[lg]
        # )
        
    # demo.queue(concurrency_count=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()