import torch
from torchtext import data
import numpy as np

from lstm import LSTMSentiment
from train_batch import load_sst
# from bert_babble.main import get_context_around

# -- SOC (Sampling and OCclusion) algorithm
#  - Parameters below following the paper's notation:
    # Input "x" of size 53
    # p = x[20:35] (the phrase which importance we want to analyze)
    # context = x[10:19] et x[35:44] for sampling
# Importance attribution: compute the differences on the scores (for positive or negative labelling) before and
# after padding, do that for every sample and them take the medium to compute the phrase importance

# Signature of the function used to sample the context (that will be imported from "bert_babble.main")
def get_context_around(seed, window_size, n_samples = 1): # (n_samples, 2)
    return [[['the'] * 10] * 2] * n_samples

# Wrapper of "get_context_around" taking into account the compatibility between the dictionaries from "bert_babble" and the one used here
def get_valid_context_around(seed, window_size, n_samples, vocabulary):
    samples_needed = n_samples
    context = [[""] * 2] * n_samples

    while samples_needed > 0:
        context_candidate = get_context_around(seed, window_size, 1)

        valid = all([word in vocabulary for word in context_candidate[0][0]]) and all([word in vocabulary for word in context_candidate[0][1]])
        if(not valid):
            print('not valid')
            continue

        # print('samples_needed:', samples_needed)
        context[samples_needed-1][0] = context_candidate[0][0]
        context[samples_needed-1][1] = context_candidate[0][1]
        samples_needed = samples_needed - 1

    return context

# Generate the model predictions and get the score attributed to the label that should be the correct one
def get_scores_for_right_class(model, text, labels):
    preds = model(text)
    # print('labels:', labels)
    # print('preds.shape:', preds.shape)
    labels_01 = labels-1
    # print('labels_01:', labels_01.shape)
    scores = preds[range(0, preds.shape[0]), labels_01]

    # print('scores.shape:', scores.shape)
    return scores # pred[label]

# Apply padding (SOC algorithm)
def apply_padding(text, start_phrase, end_phrase):
    text_padded = text.clone()
    for i in range(start_phrase, end_phrase+1):
        text_padded[i] = torch.zeros(text_padded.shape[1])

    return text_padded

# Return a single string with the words
def get_words_from_tensor(tensor, idx_to_word):
    words = []
    for i in range(tensor.shape[0]):
        idx = tensor[i]
        words.append(idx_to_word[idx])

    return " ".join(words)

# Generate the samples for the context of each phrase "p" in each input "x" (columns of "text") and return the "sampled_text"
# original "text" is a torch tensor of size [53, 1821]
def get_sampled_text(text, n_samples, start_phrase, end_phrase, window_size, text_field):
    sampled_text = torch.zeros(text.shape[0], 1, dtype=torch.long)
    for i in range(text.shape[1]):
        x = text[:, i]

        word_to_idx = text_field.vocab.stoi
        idx_to_word = text_field.vocab.itos

        seed = get_words_from_tensor(x[start_phrase:end_phrase+1], idx_to_word)

        context = get_valid_context_around(seed, window_size, n_samples, word_to_idx.keys()) # (n_samples, 2)

        for sample in range(n_samples):
            context_left = context[sample][0]
            context_right = context[sample][1]

            sampled_x = x
            for j in range(window_size):
                idx1 = start_phrase-window_size+j
                # print('j:', j)
                # print('len(context_left):', len(context_left))
                sampled_x[start_phrase-window_size+j] = word_to_idx[context_left[j]]
                sampled_x[end_phrase+1+j] = word_to_idx[context_right[j]]
            
            # print('sampled_x.shape', sampled_x.shape)
            # print('sampled_text.shape', sampled_text.shape)
            sampled_x = sampled_x.reshape(53, 1) 
            sampled_text = torch.cat((sampled_text, sampled_x), 1)

    sampled_text = sampled_text[:, 1:]
    return sampled_text

# Generated the corresponding labels for "sampled_text"
def get_sampled_labels(labels, n_samples):
    sampled_labels = torch.zeros(labels.shape[0]*n_samples, dtype=torch.long)
    
    for i in range(labels.shape[0]):
        for ii in range(n_samples):
            sampled_labels[i+ii] = labels[i]
    
    return sampled_labels

# Main function for the SOC algorithm below:
if __name__ == '__main__':
    # USE_GPU = torch.cuda.is_available()
    # EMBEDDING_DIM = 300
    # HIDDEN_DIM = 150
    BATCH_SIZE = 5
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    n_samples = 0
    start_phrase = 0 # 20
    end_phrase = 26 # 35
    window_size = 10 # for each side around the phrase

    # model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=0, label_size=0,\
    #                       use_gpu=USE_GPU, batch_size=BATCH_SIZE)
    model = torch.load('./data/modelLSTM.pt')

    train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE)

    for batch in test_iter:
        original_text = batch.text
        original_labels = batch.label
        # print('text.shape:', text.shape) # torch.Size([53, 1821])

        text = original_text
        labels = original_labels

        # print('type(text):', type(text))

        if n_samples > 0:
            text = get_sampled_text(original_text, n_samples, start_phrase, end_phrase, window_size, text_field)
            labels = get_sampled_labels(original_labels, n_samples)
            # print('text.shape:', text.shape)

        scores = get_scores_for_right_class(model, text, labels)

        padded_text = apply_padding(text, start_phrase, end_phrase)
        scores_after_padding = get_scores_for_right_class(model, padded_text, labels)

        # For example, in the case n_samples = 5
        # 0, 5, 10 ... = 5*i, i = 0 ... 1820 -> start of a new sampled block (of size n_samples)
        # for the columns of the format 5*i+k, k = 0...4 -> "x" sampled

        # In the case n_samples = 0 every column will be the start of a block (of size 1, only one "x")
        block_size = n_samples if n_samples > 0 else 1

        for i in range(int(text.shape[1]/block_size)):
            if i > 0: 
                break # stop after the first example (getting just one result for qualitative analysis)
            
            ii = i
            differences = []

            for ii in range(i, i+block_size):
                x = text[:, ii]
                label = labels[ii]

                print()
                print('Input "x" considered:')
                for pos in range(len(x)):
                    word_embedded = x[pos].item()
                    idx_to_word = text_field.vocab.itos
                    word_to_idx = text_field.vocab.stoi
                    # print('word_to_idx[" "]:', word_to_idx[" "])
                    print(pos, '', idx_to_word[word_embedded])

                print('label (1 is positive, 2 is negative):', label.item()) # 1 is positive, 2 is negative
                print('Phrase "p" from position', start_phrase, 'to', end_phrase)
                print('scores[ii]:', scores[ii].item())
                print('scores_after_padding[ii]:', scores_after_padding[ii].item())

                difference = scores[ii].item() - scores_after_padding[ii].item()
                differences.append(difference)

                print('difference:', difference)
                #get_scores(model, x, label)

            mean_difference = np.array(differences).mean()
            
            # "mean_difference" measures the importance of the phrase "p" (expected to always 
            # be positive as it is already the score contribution to the correct label)
            print(i, ') mean_difference (phrase importance): ', mean_difference, sep='') 

            print()


