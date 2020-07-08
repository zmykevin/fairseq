import torch
import os
import json
from tqdm import tqdm

#Load the translation model
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',tokenizer='moses', bpe='subword_nmt')
en2de.eval()
en2de.cuda()
print("Loaded the Translation Model")

#Load the original CC data
data_path = "/home/zmykevin/multilingual_VL/data/CC/annotations"
cc_annotation = '/'.join([data_path, "dataset_cc.json"])

with open(cc_annotation, "r") as f:
    cc_data = json.load(f)
print("Loaded CC Original Data. \n")

#Translate the CC
cc_imgs = cc_data['images']
cc_data_de = {"images":[], "dataset": cc_data['dataset']}
for img in tqdm(cc_imgs):
    img_de = img.copy()
    #translate the caption
    caption_en = img_de['sentences'][0]['raw']
    caption_de = en2de.translate(caption_en) 
    tokens_de = caption_de.split()
    #update img_de
    img_de['sentences'][0]['tokens'] = tokens_de
    img_de['sentences'][0]['raw'] = caption_de
    cc_data_de['images'].append(img_de)
    #break

print("Finished Translation. \n")

#save the new data to output file
output_cc_data_path = '/'.join([data_path, "dataset_cc_de.json"])
with open(output_cc_data_path, "w") as f_out:
    json.dump(cc_data_de, f_out, indent=4, sort_keys=True)


