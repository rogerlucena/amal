import torch
from torchtext import data
import numpy as np

from lstm import LSTMSentiment
from train_batch import load_sst
# from bert_babble.main import get_context_around

# -- SOC -- 
# input "x" of size 53
# p = x[20:35]
# context = x[10:19] et x[35:44] for sampling
# compute the differences on the scores and take the medium of them! =D

# -- SCD --
# find on github a working CD (Contextual Decomposition)
# apply sampling as in SOC 

def get_context_around(seed, window_size, n_samples = 1): # (n_samples, 2)
    return [[['the'] * 10] * 2] * n_samples

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

def get_scores_for_right_class(model, text, labels):
    preds = model(text)
    # print('labels:', labels)
    # print('preds.shape:', preds.shape)
    labels_01 = labels-1
    # print('labels_01:', labels_01.shape)
    scores = preds[range(0, preds.shape[0]), labels_01]

    print('scores.shape:', scores.shape)
    return scores # pred[label]

def apply_padding(text):
    text_padded = text.clone()
    for i in range(45):
        text_padded[i] = torch.zeros(text_padded.shape[1])

    return text_padded

# return a single string with the words
def get_words_from_tensor(tensor, idx_to_word):
    words = []
    for i in range(tensor.shape[0]):
        idx = tensor[i]
        words.append(idx_to_word[idx])

    return " ".join(words)

# text is a torch tensor of size [53, 1821]
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

def get_sampled_labels(labels, n_samples):
    sampled_labels = torch.zeros(labels.shape[0]*n_samples, dtype=torch.long)
    
    for i in range(labels.shape[0]):
        for ii in range(n_samples):
            sampled_labels[i+ii] = labels[i]
    
    return sampled_labels

if __name__ == '__main__':

    # USE_GPU = torch.cuda.is_available()
    # EMBEDDING_DIM = 300
    # HIDDEN_DIM = 150
    BATCH_SIZE = 5
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    n_samples = 1
    start_phrase = 20
    end_phrase = 35
    window_size = 10 # for each side around phrase

    # model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=0, label_size=0,\
    #                       use_gpu=USE_GPU, batch_size=BATCH_SIZE)

    model = torch.load('./data/modelLSTM.pt')

    train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE)

    for batch in test_iter:
        text = batch.text
        labels = batch.label
        # print('text.shape:', text.shape) # torch.Size([53, 1821])

        # print('type(text):', type(text))
        sampled_text = get_sampled_text(text, n_samples, start_phrase, end_phrase, window_size, text_field)
        sampled_labels = get_sampled_labels(labels, n_samples)
        # print('sampled_text.shape:', sampled_text.shape)

        scores = get_scores_for_right_class(model, sampled_text, sampled_labels)

        padded_text = apply_padding(sampled_text)
        scores_after_padding = get_scores_for_right_class(model, padded_text, sampled_labels)

        # 0, 5, 10 ... = 5*i, i = 0 ... 1820 -> start of a new sampled block
        # for the columns of the format 5*i+k, k = 0...4 -> "x" sampled
        for i in range(int(text.shape[1]/n_samples)):
            if i > 1:
                break
            
            ii = i
            differences = []

            for ii  in range(i, i+n_samples):
                x = sampled_text[:, ii]
                label = sampled_labels[ii]

                for pos in range(len(x)):
                    word_embedded = x[pos].item()
                    idx_to_word = text_field.vocab.itos
                    word_to_idx = text_field.vocab.stoi
                    # print('word_to_idx[" "]:', word_to_idx[" "])
                    print(pos, '', idx_to_word[word_embedded])

                print('scores[ii]:', scores[ii].item())
                print('scores_after_padding[ii]:', scores_after_padding[ii].item())

                difference = scores[ii].item() - scores_after_padding[ii].item()
                differences.append(difference)

                print('difference:', difference)
                #get_scores(model, x, label)

            mean_difference = np.array(differences).mean()
            print(i, '- mean_difference:', mean_difference)

            # pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
            # pretrained_embeddings[0] = 0
            # word2vec = load_bin_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
            # for word, vector in word2vec.items():
            #     pretrained_embeddings[word_to_idx[word]-1] = vector

    
            # print('len(x):', len(x))

            # start = int (0.2 * size)
            # end = int (0.7 * size)

            # p = x # x[start:end]

            # print('x:', x)
            # print('p:', p)
            # print('type(x)', type(x))

            # print('x[0].shape:', x[0].shape) # torch.Size([53, 1821])
            # print('x[1].shape:', x[1].shape) # torch.Size([1821])


