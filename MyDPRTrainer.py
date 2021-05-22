import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomDPRDataset import CustomDPRDataset
from tqdm import tqdm
import sys

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AdamW, get_linear_schedule_with_warmup

# initialize tokenizers and models for context encoder and question encoder
context_name = 'facebook/dpr-ctx_encoder-multiset-base' # set to what context encoder we want to use
question_name = 'facebook/dpr-question_encoder-multiset-base' # set to what question encoder we want to use
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_name)
context_model = DPRContextEncoder.from_pretrained(context_name).to("cuda:0")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_name)
question_model = DPRQuestionEncoder.from_pretrained(question_name).to("cuda:0")

nll = nn.NLLLoss()
# question_model.half()
# context_model.half()


# params
batch_size = 256
microbatchsize = 15
grad_accum = 24
lr = 1e-5
text_descrip = "batchsize256_gradaccum8_v2"

print("intialized models/tokenizers")

# initialize dataset
train_dataset = CustomDPRDataset()

print("created dataset")

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

    # query embeddings
    question = query.split('///')[0]
    query_input_ids = question_tokenizer(question, return_tensors='pt').to("cuda:0")
    query_embeddings = question_model(**query_input_ids).pooler_output.cuda()

    #highest doc/lowest doc embeddings
    highest_doc_input_ids = context_tokenizer(highest_doc, return_tensors='pt').to("cuda:0")
    highest_doc_embeddings = context_model(**highest_doc_input_ids).pooler_output.cuda()
    lowest_doc_input_ids = context_tokenizer(lowest_doc, return_tensors='pt').to("cuda:0")
    lowest_doc_embeddings = context_model(**lowest_doc_input_ids).pooler_output.cuda()

    # other gold doc embeddings
    other_gold_doc_embeddings = [highest_doc_embeddings, lowest_doc_embeddings]
    # print("other_gold_docs: " , len(other_gold_docs))
    for i in range(0, len(other_gold_docs), microbatchsize):
        cur_docs = other_gold_docs[i:min(i+microbatchsize, len(other_gold_docs))]
        gold_doc_toks = context_tokenizer.batch_encode_plus(cur_docs, return_tensors='pt', truncation=True, padding=True).to("cuda:0")
        gold_doc_embeddings = context_model(**gold_doc_toks).pooler_output
        other_gold_doc_embeddings.extend(list(torch.split(gold_doc_embeddings, gold_doc_embeddings.shape[0])))
    other_gold_doc_embeddings = torch.cat(other_gold_doc_embeddings, dim=0)
    #other_gold_doc_embeddings = other_gold_doc_embeddings.view(-1, other_gold_doc_embeddings.shape[-1])

    # make matrix of query embedding
    query_embeddings_arr = []
    for i in range(batch_size):
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)
        num_training_steps_scheduler = len(train_dataset)
        num_warmup_steps_scheduler = 5
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps_scheduler, num_training_steps_scheduler)
        # loop through all values in the dataset
        prev_loss = None
        avg_loss = None
        for idx in tqdm(range(len(train_dataset)), total=len(train_dataset), desc="running through training data"):
            #print(len(train_dataset[idx]))
            curr = train_dataset[idx]
            query, high_d, low_d, other_d = curr[0], curr[1], curr[2], curr[3]
            loss = traversal_for_question_context(query, high_d, low_d, other_d)
            loss.backward()
            if idx < grad_accum:
                total_loss += loss.detach().cpu().item()
            else:
                total_loss += loss.detach().cpu().item()
            if (idx+1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                # zero the gradient
                question_model.zero_grad()
                context_model.zero_grad()
            prev_loss = loss.detach()
            if idx % see_loss_step == 0:
                total_loss /= see_loss_step
                loss_to_print = total_loss
                #print(total_loss.item())
                print("Loss within curr epoch at " + str(idx) + ":", loss_to_print)
                # save model
                total_loss = 0.0
            if idx % save_model_step == 0:
                print("saving model at epoch: " + str(i))
                question_model_curr_save = 'saved_question_models_'+text_descrip+'/model_' + str(i)
                context_model_curr_save = 'saved_context_models_'+text_descrip+'/model_' + str(i)
                question_model.save_pretrained(question_model_curr_save)
                context_model.save_pretrained(context_model_curr_save)
        #loss_to_print = avg_loss if avg_loss else total_loss
        #print("Loss after epoch " + str(i) + ": ", loss_to_print.detach())

# CALL FUNCTION
train_dpr()
