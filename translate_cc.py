import torch
import os
import json
from tqdm import tqdm
import time

#Load the translation model
#en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',tokenizer='moses', bpe='subword_nmt')
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',tokenizer='moses', bpe='fastbpe',checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt')
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



total_elements = len(cc_imgs)
batch_size = 32
cc_data_de = {"images":[], "dataset": cc_data['dataset']}
output_cc_data_path = '/'.join([data_path, "dataset_cc_de.json"])

start_time = time.perf_counter()
for i in tqdm(range(0,total_elements,batch_size)):
    #form the batch of sentences
    captions_en_batch = [x['sentences'][0]['raw'] for x in cc_imgs[i:i+batch_size]]
    #print(captions_en_batch)
    captions_de_batch = en2de.translate(captions_en_batch)
    #print(captions_de_batch)
    for j in range(i,i+batch_size):
        cc_data['images'][j]['sentences'][0]['raw'] = captions_de_batch[j-i]
        cc_data['images'][j]['sentences'][0]['tokens'] = captions_de_batch[j-i].split()
        cc_data_de['images'].append(cc_data['images'][j])
    #save cc_data_de
    if i % 32000 == 0:
        with open(output_cc_data_path, "w") as f_out:
            json.dump(cc_data_de, f_out, indent=4, sort_keys=True)
        #break
    # if i > 64:
    # 	break
#print(cc_data_de["images"][0])
print("Finished Translation in {} seconds. \n".format(time.perf_counter()-start_time))
# cc_data_de = {"images":[], "dataset": cc_data['dataset']}
# for img in tqdm(cc_imgs):
#     img_de = img.copy()
#     #translate the caption
#     caption_en = img_de['sentences'][0]['raw']
#     caption_de = en2de.translate(caption_en) 
#     tokens_de = caption_de.split()
#     #update img_de
#     img_de['sentences'][0]['tokens'] = tokens_de
#     img_de['sentences'][0]['raw'] = caption_de
#     cc_data_de['images'].append(img_de)
#     #break



# #save the new data to output file
# output_cc_data_path = '/'.join([data_path, "dataset_cc_de.json"])
# with open(output_cc_data_path, "w") as f_out:
#     json.dump(cc_data_de, f_out, indent=4, sort_keys=True)


