import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBart50TokenizerFast,MBartForConditionalGeneration,AutoTokenizer,AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
import glob

def translation(df,lg):
    # from transformers import MarianTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                sentence1 = df[index]["target_caption"] # 这里target_caption是其他语言
                # sentence1 = df[index]
                # sentence1 = df[index]["caption"]

                tokens = self.tokenizer(sentence1, return_tensors="pt",truncation=True,max_length=75)
            except Exception as e:
                print(e)
                sentence = "This is a default caption."
                tokens = self.tokenizer(sentence, return_tensors="pt",truncation=True,max_length=75)
            return tokens


    print("model loading..")
    # model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("/share/project/yfl/database/hub/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/0ece2bb75a89350002537169ecadeb2b3d043b6b")
    tokenizer = MBart50TokenizerFast.from_pretrained("/share/project/yfl/database/hub/models--facebook--mbart-large-50-many-to-many-mmt/snapshots/0ece2bb75a89350002537169ecadeb2b3d043b6b")
    # tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    # model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    print("model loading completed.")

    # translate Hindi to French
    tokenizer.src_lang = lg
    # encoded_hi = tokenizer(article_hi, return_tensors="pt")
    # generated_tokens = model.generate(
    #     **encoded_hi,
    #     forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    # )
    # tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    model.to(device)
    model.eval()


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
        batch_size=196,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    )

    with torch.no_grad():
        decoded_tokens = []
        for i, batch in enumerate(tqdm(test_dataloader)):

            batch = {k: v.to(device) for k, v in batch.items()}
            output_tokens = model.generate(**batch,max_length=75,num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
            # output_tokens = model.generate(**batch,max_length=256,num_beams=5, )
            decoded_tokens = tokenizer.batch_decode(output_tokens.to("cpu"), skip_special_tokens=True)
            for j, target_caption in enumerate(decoded_tokens):
                # 糟了 这里之前好像出问题了 噢~没有问题这是我写的
                # df[i*196+j]['target_caption'] = target_caption
                df[i*196+j]['source_caption'] = target_caption # 这里是把其他语言翻译成英文
                # df[i*196+j]['target'] = lg
    # df["target_text"] = decoded_tokens
    # df['target'] = lg

    return df
if __name__=='__main__':
    # import sys
    # path,lg = sys.argv[1],sys.argv[2]
    # print(path)
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--lg', type=str)
    
    arg = parser.parse_args()
    lg = arg.lg
    
    print("Loading...")
    with open(arg.path,"r") as f:
        df = json.load(f)
    print("Loaded ^_^ !")
#     df = pd.read_json(path,lines=True)
    result = translation(df,lg)
    
    # save_path = arg.path.rsplit('.', 1)[0] + '_' + arg.lg + '.json'
    save_path = arg.path # 覆盖原来的文件即可；
    print("save in {}".format(save_path))
    with open(save_path, 'w') as fn:
        json.dump(result, fn)
    # import pdb
    # pdb.set_trace()
    print("Fnished!")
    # df.to_json(path + f"_{lg}",orient='records',lines=True)
    # with open('/sharefs/baai-mrnd/yfl/codebase/nb/fr.json', 'w') as fn:
    #     json.dump(decode, fn)
