import os
import json
import argparse
import pandas as pd

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MBart50TokenizerFast,MBartForConditionalGeneration,AutoTokenizer,AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from glob import glob

# lg2code_dict = {
#     "en": "en_XX", 
#     "zh": "zh_CN",
#     "ar": "ar_AR",
#     "de": "de_DE",
#     "es": "es_XX",
#     "fr": "fr_XX",
#     "hi": "hi_IN",
#     "it": "it_IT",
#     "ja": "ja_XX",
#     "ko": "ko_KR",
#     "nl": "nl_XX",
#     "pl": "pl_PL",
#     "pt": "pt_XX",
#     "ru": "ru_RU",
#     "th": "th_TH",
#     "tr": "tr_TR",
#     "uk": "uk_UA",
#     "vi": "vi_VN",
# }

lg2code_dict = {
    "en": "eng_Latn", 
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
}


def translation(df,model, tokenizer, lg):
    # from transformers import MarianTokenizer

    class CaptionDataset(Dataset):
        def __init__(self, df, tokenizer):
            self.df = df
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            # sentence1 = df.loc[index, "source_text"]
            try:
                # sentence1 = df[index]["source_caption"]
                sentence1 = df[index] # 这里target_caption是其他语言
                # sentence1 = df[index]
                # sentence1 = df[index]["caption"]

                tokens = self.tokenizer(sentence1, return_tensors="pt",truncation=True,max_length=75)
            except Exception as e:
                print(e)
                sentence = "This is a default caption."
                tokens = self.tokenizer(sentence, return_tensors="pt",truncation=True,max_length=75)
            return tokens


    def custom_collate_fn(data):
        """
        Data collator with padding.
        """
        tokens = [sample["input_ids"][0] for sample in data]
        attention_masks = [sample["attention_mask"][0] for sample in data]

        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
        padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)

        batch = {"input_ids": padded_tokens, "attention_mask": attention_masks}
        return batch


    test_data = CaptionDataset(df, tokenizer)
    test_dataloader = DataLoader(
        test_data,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    new_df = []
    
    with torch.no_grad():
        decoded_tokens = []
        for i, batch in enumerate(tqdm(test_dataloader)):

            batch = {k: v.to(device) for k, v in batch.items()}
            output_tokens = model.generate(**batch,max_length=75,num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
            # output_tokens = model.generate(**batch,max_length=75,num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"])
            # output_tokens = model.generate(**batch,max_length=75,num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"])
            # output_tokens = model.generate(**batch,max_length=256,num_beams=5, )
            decoded_tokens = tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)
            for j, target_caption in enumerate(decoded_tokens):
                # 糟了 这里之前好像出问题了 噢~没有问题这是我写的
                # df[i*196+j]['target_caption'] = target_caption
                # df[i*10+j]['source_caption'] = target_caption # 这里是把其他语言翻译成英文
                new_df.append(target_caption)
                # df[i*196+j]['target'] = lg
    # df["target_text"] = decoded_tokens
    # df['target'] = lg

    return new_df
if __name__=='__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str)
    # parser.add_argument('--lg', type=str)
    
    # arg = parser.parse_args()
    save_path = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/translated_prompts'
    input_dir = '/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/prompts/*'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model loading..")
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # tokenizer = AutoTokenizer.from_pretrained("/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/cache/nllb-200-distilled-600M", src_lang=lg)
    # model = AutoModelForSeq2SeqLM.from_pretrained("/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/cache/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/cache/nllb-200-3.3B")
    # model = MBartForConditionalGeneration.from_pretrained("/share/project/yfl/database/hub/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/0ece2bb75a89350002537169ecadeb2b3d043b6b")
    # tokenizer = MBart50TokenizerFast.from_pretrained("/share/project/yfl/database/hub/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/0ece2bb75a89350002537169ecadeb2b3d043b6b")
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    print("model loading completed.")
    
    # translate Hindi to French
    # tokenizer.src_lang = lg
    # encoded_hi = tokenizer(article_hi, return_tensors="pt")
    # generated_tokens = model.generate(
    #     **encoded_hi,
    #     forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    # )
    # tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    model.to(device)
    model.eval()
    
    file_names = glob(input_dir)

    
    for file_name in file_names:
        lg = file_name.rsplit('_', 1)[1].split('.')[0]
        lg_code = lg2code_dict[lg]
        # tokenizer = AutoTokenizer.from_pretrained("/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/evaluation_new/cache/nllb-200-distilled-600M", src_lang=lg)
        tokenizer = AutoTokenizer.from_pretrained("/share/project/yfl/codebase/git/AltTools/Altdiffusion/misc/human_evaluation/cache/nllb-200-3.3B", src_lang=lg_code)
        # tokenizer = MBart50TokenizerFast.from_pretrained("/share/project/yfl/database/hub/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/0ece2bb75a89350002537169ecadeb2b3d043b6b", src_lang=lg_code)
    
        # opt.save_path = save_path+ '/' + lg
        # if not os.path.exists(opt.save_path):
        #     os.makedirs(opt.save_path)
        
        df = []
        with open(file_name, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                df.append(line)

        lines = translation(df, model, tokenizer, lg)
        
        final_save_path = save_path + '/' + 'prompt_' + lg + '.txt'
        with open(final_save_path, 'w') as file:
            
            for line in lines:
                file.write(line + '\n')
    
