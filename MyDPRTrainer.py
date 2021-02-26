import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomDPRDataset import CustomDPRDataset
from tqdm import tqdm
import sys

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW

# initialize tokenizers and models for context encoder and question encoder
context_name = 'facebook/dpr-ctx_encoder-multiset-base' # set to what context encoder we want to use
question_name = 'facebook/dpr-question_encoder-multiset-base' # set to what question encoder we want to use
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_name)
context_model = DPRContextEncoder.from_pretrained(context_name).cuda()
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_name)
question_model = DPRQuestionEncoder.from_pretrained(question_name).cuda()
nll = nn.NLLLoss()

print("intialized models/tokenizers")

# initialize dataset
train_dataset = CustomDPRDataset()

print("created dataset")

# add tokens to question encoder
tokens_to_add = ['<Enables>', '<Motivates>', '<Results-in>'] # from the glucose connectives
question_tokenizer.add_tokens(tokens_to_add) #  add tokens
question_model.question_encoder.bert_model.resize_token_embeddings(len(question_tokenizer)) # resize model to fit new tokens

print("added tokens")

def traversal_for_question_context(query, highest_doc, lowest_doc, other_gold_docs):
    '''
    parameters:
    * query -- question being asked
    * highest_doc -- positive sample (obtained from highest rouge rank)
    * lowest_doc -- negative sample (obtained from lowest rouge rank)
    * other_gold_docs -- other negative samples (gold standard of other queries)
    return:
    * loss calculated by:
    1. embed the docs (with context model)
    2. embed the query (with question model)
    3. make matrix with query of size len(other_gold_docs) + 2 
    4. take dot product of matrix from #3 with matrix holding each doc embedding
    5. softmax resulting vector
    6. NLL the 0th component (positive example)
    7. return the loss
    '''
    # zero the gradient
    question_model.zero_grad()
    context_model.zero_grad()

    # query embeddings
    query_input_ids = question_tokenizer(query, return_tensors='pt')["input_ids"].cuda()
    query_embeddings = question_model(query_input_ids).pooler_output.cuda()

    #highest doc/lowest doc embeddings
    highest_doc_input_ids = context_tokenizer(highest_doc, return_tensors='pt')["input_ids"].cuda()
    highest_doc_embeddings = context_model(highest_doc_input_ids).pooler_output.cuda()
    lowest_doc_input_ids = context_tokenizer(lowest_doc, return_tensors='pt')["input_ids"].cuda()
    lowest_doc_embeddings = context_model(lowest_doc_input_ids).pooler_output.cuda()

    # other gold doc embeddings
    other_gold_doc_embeddings = [highest_doc_embeddings, lowest_doc_embeddings]
    for gold_doc in other_gold_docs:
        gold_doc_input_ids = context_tokenizer(gold_doc, return_tensors='pt')["input_ids"].cuda()
        gold_doc_embeddings = context_model(gold_doc_input_ids).pooler_output
        other_gold_doc_embeddings.append(gold_doc_embeddings)
    other_gold_doc_embeddings = torch.cat(other_gold_doc_embeddings)

    # make matrix of query embedding
    query_embeddings_arr = []
    for i in range(16):
        query_embeddings_arr.append(query_embeddings)
    query_embeddings_matrix = torch.cat(query_embeddings_arr).cuda()

    # take dot product
    matrix_mult = torch.matmul(query_embeddings_matrix, other_gold_doc_embeddings.T)
    dot_product = torch.diagonal(matrix_mult, 0)

    # take softmax
    softmax = F.softmax(dot_product)

    #NLL of the 0th component
    #loss = nll(softmax[0])
    loss = -torch.log(softmax[0])

    return loss


def train_dpr(num_epochs=1, see_loss_step=50, save_model_step=500):
    # loop through all epochs
    for i in range(num_epochs):
        print("EPOCH: ", i)
        # set to train
        total_loss = 0.0
        question_model.train()
        context_model.train()
        # create optimizer
        optimizer_grouped_parameters = [
            {'params': [p for n, p in question_model.named_parameters()]},
            {'params': [p for n, p in context_model.named_parameters()]}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        # loop through all values in the dataset
        for idx in tqdm(range(len(train_dataset)), total=len(train_dataset), desc="running through training data"):
            #print(len(train_dataset[idx]))
            for curr in train_dataset[idx]:
                query, high_d, low_d, other_d = curr[0], curr[1], curr[2], curr[3]
                loss = traversal_for_question_context(query, high_d, low_d, other_d)
                total_loss += loss
                loss.backward()
                optimizer.step()
                if idx % see_loss_step == 0:
                    print("Loss within curr epoch at " + str(idx) + ":", total_loss.detach())
                 # save model
                if idx % save_model_step == 0:
                    print("saving model at epoch: " + str(i))
                    question_model_curr_save = 'saved_question_models/model_' + str(i)
                    context_model_curr_save = 'saved_context_models/model_' + str(i)
                    question_model.save_pretrained(question_model_curr_save)
                    context_model.save_pretrained(context_model_curr_save)

        print("Loss after epoch " + str(i) + ": ", total_loss.detach())

# CALL FUNCTION
train_dpr()