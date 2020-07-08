import torch

#List available models
torch.hub.list('pytorch/fairseq')

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',tokenizer='moses', bpe='subword_nmt')
en2de.eval()

en2de.cuda()

# Translate a sentence
print(en2de.translate('Hello world!'))
# 'Hallo Welt!'