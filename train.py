from fastai import *
from fastai.text import *
from fastai.text.all import *
import pandas as pd
import pickle

import fastai
import torch
print(f"torch version: {torch.__version__}")
print(f"fastai version: {fastai.__version__}")
print(f"GPU which is used : {torch.cuda.get_device_name(0)}")

## parameters for dataloader and tokenizer
lang = "fa"
backwards=False
bs=128
vocab_sz = 30000
drop_mult = 0.5
num_workers=18
## setting pathes
base = Path(".").absolute()
spm_path = base / "spm"
out = base / "model_out"
lm_fns = [out / f"{lang}_ULMFIT", out / f"{lang}_ULMFIT_vocab.pkl"]
spm_path.mkdir(exist_ok=True)
out.mkdir(exist_ok=True)

## reading train csv data
df = pd.read_csv("data_ULMFIT.csv")

## using sentence piece tokenizer
tok = SentencePieceTokenizer(lang=lang, max_vocab_sz=vocab_sz, cache_dir=spm_path)

dblock = DataBlock(
    blocks=(TextBlock.from_df('text', is_lm=True, tok=tok,backwards=backwards)),
    get_x=ColReader('text'))


dls = dblock.dataloaders(sample, bs=bs)

learn = language_model_learner(dls, AWD_LSTM, drop_mult=0.1, wd=0.1, pretrained=False,cbs=[
                CSVLogger(fname=base/"history.csv")],metrics=[accuracy,Perplexity()]).to_fp16()

lr = 2e-4
lr *= bs/48  # Scale learning rate by batch size
num_epochs = 10
## fitting the model
learn.unfreeze()
learn.fit_one_cycle(num_epochs, lr,moms=(0.8, 0.7, 0.8))

## saving pretrained model
with open(lm_fns[1], 'wb') as f:
      pickle.dump(learn.dls.vocab, f)

learn.to_fp32().save(lm_fns[0],with_opt=False)
stats = learn.recorder.values[-1]

train_params = {
    'lang': lang,
    'backwards': backwards,
    'batch_size': bs,
    'lr': lr,
    'num_epochs': num_epochs,
    'drop_mult': drop_mult,
    'tokenizer': {
        'class': tok.__class__.__name__,
        'params': {
            'lang': lang,
            'vocab_sz': vocab_sz
        }
    },
    'stats': {
        'train_loss': stats[0],
        'valid_loss': stats[1],
        'accuracy': stats[2],
        'perplexity': stats[3]
    }
}

with open('model.json', 'w') as f:
    json.dump(train_params, f, ensure_ascii=False, indent=4)
