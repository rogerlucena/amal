import math
import time

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model (weights)
model_version = 'bert-base-uncased'
print("Loading pre-trained model")
model = BertForMaskedLM.from_pretrained(model_version)
model.eval()
cuda = torch.cuda.is_available()
if cuda:
    model = model.cuda()

# Load pre-trained model tokenizer (vocabulary)
print("Loading vocabulary")
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=model_version.endswith("uncased"))

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx
  
  
def get_init_text(seed_text, max_len, batch_size = 1):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [ [MASK] * max_len + seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    return tokenize_batch(batch)

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))

# Generation modes as functions

def parallel_sequential_generation(seed_text, batch_size=10, mask_len=14, top_k=0, temperature=None, max_iter=300, burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep
    
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, mask_len, batch_size)
    
    for ii in range(max_iter):
        kk = np.random.randint(0, mask_len) if np.random.randint(0,2) == 0 else np.random.randint(seed_len + mask_len, seed_len + 2* mask_len)
        for jj in range(batch_size):
            batch[jj][kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        idxs = idxs if hasattr(idxs, "__getitem__") else [idxs]
        for jj in range(batch_size):
            batch[jj][kk] = idxs[jj]
            
        if verbose and np.mod(ii+1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:kk+1] + ['(*)'] + for_print[kk+1:]
            print("iter", ii+1, " ".join(for_print))
            
    return untokenize_batch(batch)

def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25, 
             generation_mode="parallel-sequential",
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, mask_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                                   cuda=cuda, verbose=False)
        else:
            raise Exception("Invalid Generation Mode")
        
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()
        
        sentences += batch
    return sentences

def get_context_around(seed="countryside around her", window_size=10, samples=1):
    # Parameters
    n_samples = samples
    batch_size = samples
    mask_len = window_size + 1
    top_k = 100
    temperature = 1.0
    generation_mode = "parallel-sequential"
    leed_out_len = 5 # max_len
    burnin = 250
    sample = True
    max_iter = 500
    seed_text = seed.split()
    bert_sents = generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=mask_len,
                      generation_mode=generation_mode,
                      sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                      cuda=cuda)

    results = []
    for sent in bert_sents:
        detokenized = detokenize(sent)[1:-1]
        results.append([detokenized[0:window_size], detokenized[(window_size + len(seed_text)):]])
    return results

if __name__ == "__main__":
    print("Calculating context")
    print(get_context_around())