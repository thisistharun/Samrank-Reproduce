import json
import argparse
import logging
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

import torch

from transformers import AutoTokenizer, BertModel, AutoModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import RobertaTokenizer, RobertaModel

from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates


# Run "stanford-corenlp-full-2018-02-27" with terminal before run this code.
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &

logging.basicConfig(filename='error_log.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Initialize POS tagger
host = 'localhost'
port = 9000
pos_tagger = PosTaggingCoreNLP(host, port)

# Load stopwords
def load_stopwords(filepath):
    with open(filepath, "r") as file:
        return [line.strip() for line in file if line.strip()]

stopwords = load_stopwords('UGIR_stopwords.txt')

# Initialize stemmer
stemmer = PorterStemmer()

def read_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Define functions for processing attention maps
def get_col_sum_token_level(attention_map):
    return torch.sum(attention_map, dim=0)

def redistribute_global_attention_score(attention_map, tokens_score):
    return attention_map * tokens_score.unsqueeze(0)

def normalize_attention_map(attention_map):
    attention_map_sum = attention_map.sum(dim=0, keepdim=True) + 1e-10
    return attention_map / attention_map_sum

def get_row_sum_token_level(attention_map):
    return torch.sum(attention_map, dim=1)

# Define functions for processing phrases
def aggregate_phrase_scores(index_list, tokens_scores):
    return sum(tokens_scores[start:end].sum() for start, end in index_list)

def get_phrase_indices(text_tokens, phrase, prefix):
    clean_tokens = [token.replace(prefix, '') for token in text_tokens]
    phrase = phrase.replace(' ', '')
    matched_indices = []
    target = phrase
    i = 0
    while i < len(clean_tokens):
        if clean_tokens[i].startswith(target):
            start = i
            while i < len(clean_tokens) and target:
                length = len(clean_tokens[i])
                target = target[length:]
                i += 1
            matched_indices.append((start, i))
            target = phrase
        else:
            i += 1
    return matched_indices

# Define functions for handling candidates
def remove_repeated_sub_word(candidates_pos_dict):
    """
    Remove positions of single words if they are fully contained within any of the positions of longer phrases.
    """
    for phrase in list(candidates_pos_dict.keys()):
        if ' ' in phrase:  # This ensures we're only checking multi-word phrases
            phrase_positions = candidates_pos_dict[phrase]
            for word in phrase.split():
                if word in candidates_pos_dict:
                    single_word_positions = candidates_pos_dict[word]
                    # Filter positions where the single word is not fully contained within any multi-word phrase positions
                    filtered_positions = [pos for pos in single_word_positions if not any(p_start <= pos[0] and p_end >= pos[1] for p_start, p_end in phrase_positions)]
                    candidates_pos_dict[word] = filtered_positions

    return candidates_pos_dict


def get_same_len_segments(total_tokens_ids, max_len):
    num_of_seg = (len(total_tokens_ids) + max_len - 1) // max_len
    segments = [total_tokens_ids[i * max_len:(i + 1) * max_len] for i in range(num_of_seg)]
    attn_masks = [[1] * len(segment) for segment in segments]
    return segments, attn_masks

# Define functions for evaluating keyphrase extraction
def get_candidates(core_nlp, text):
    tagged_text = core_nlp.pos_tag_raw_text(text)
    text_obj = InputTextObj(tagged_text, 'en')
    return extract_candidates(text_obj)

def get_score_full(candidates, references, maxDepth=15):
    reference_set = set(references)
    true_positive = 0
    precision, recall = [], []
    for i, candidate in enumerate(candidates):
        if candidate in reference_set:
            true_positive += 1
        if i < maxDepth:
            precision.append(true_positive / (i + 1))
            recall.append(true_positive / len(reference_set))
    return precision, recall

def evaluate(candidates, references):
    results = {}
    precision_scores, recall_scores, f1_scores = defaultdict(list), defaultdict(list), defaultdict(list)
    for candidate, reference in zip(candidates, references):
        p, r = get_score_full(candidate, reference)
        for depth in [5, 10, 15]:
            precision = p[depth - 1] if depth <= len(p) else p[-1]
            recall = r[depth - 1] if depth <= len(r) else r[-1]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            precision_scores[depth].append(precision)
            recall_scores[depth].append(recall)
            f1_scores[depth].append(f1)

    # Print results
    print("Metrics")
    for depth in [5, 10, 15]:
        avg_precision = np.mean(precision_scores[depth])
        avg_recall = np.mean(recall_scores[depth])
        avg_f1 = np.mean(f1_scores[depth])
        print(f"@{depth}: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")
        results[f'precision@{depth}'] = avg_precision
        results[f'recall@{depth}'] = avg_recall
        results[f'f1@{depth}'] = avg_f1

    return results

# Define error logging for evaluations
def log_error_instances(candidate_set, reference_set, data_id):
    true_positive = set(candidate_set) & set(reference_set)
    false_positive = set(candidate_set) - set(reference_set)
    false_negative = set(reference_set) - set(candidate_set)
    error_details = {
        'data_id': data_id,
        'false_positive': list(false_positive),
        'false_negative': list(false_negative),
        'true_positive': list(true_positive)
    }
    logging.info(f"Error details for data ID {data_id}: {error_details}")
    return error_details


def evaluate_all_heads(layer_head_predicted_top15, dataset):
    experiment_results = []

    # Iterate through each layer and head
    for (layer, head), predicted_phrases in layer_head_predicted_top15.items():
        # Prepare gold keyphrases and predicted keyphrases lists for evaluation
        gold_keyphrase_list = [[key.lower() for key in item['keyphrases']] for item in dataset]
        predicted_keyphrase_list = [[phrase.lower() for phrase in phrases] for phrases in predicted_phrases]

        # Log the current layer and head
        print(f"Layer {layer + 1}, Head {head + 1} results:")
        
        # Evaluate the predicted keyphrases against the gold standard
        total_score = evaluate(predicted_keyphrase_list, gold_keyphrase_list)
        total_score.update({'layer': layer + 1, 'head': head + 1})
        
        # Append the results for this layer and head
        experiment_results.append(total_score)

    # Convert results to a DataFrame for further processing
    df = pd.DataFrame(experiment_results)

    # Define the path for saving results and ensure the directory exists
    results_path = f'experiment_results/{args.dataset}/'
    os.makedirs(results_path, exist_ok=True)
    
    # Save the DataFrame to CSV
    df.to_csv(f'{results_path}{args.plm}_{args.mode}.csv', index=False)

    # Extract the top 3 results based on F1 scores at different cut-offs
    top3_f1_5 = df.nlargest(3, 'f1@5').reset_index(drop=True)
    top3_f1_10 = df.nlargest(3, 'f1@10').reset_index(drop=True)
    top3_f1_15 = df.nlargest(3, 'f1@15').reset_index(drop=True)

    # Return the top results
    return top3_f1_5, top3_f1_10, top3_f1_15


def rank_short_documents(args, dataset, model, tokenizer):
    # Define token prefixes for different models
    prefixes = {
        'BERT': '##',
        'mBERT': '##',
        'GPT2': 'Ġ',
        'RoBERTa': 'Ġ',
        'XLM-R': 'Ġ'
    }
    
    prefix = prefixes.get(args.plm, '##')  # Default to '##' if model not in dictionary

    layer_head_predicted_top15 = defaultdict(list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for data in tqdm(dataset, desc="Processing documents"):
        with torch.no_grad():
            tokenized_text = tokenizer(data['text'], return_tensors='pt', max_length=512, truncation=True)
            outputs = model(**tokenized_text.to(device))
            attentions = outputs.attentions

            candidates = get_candidates(pos_tagger, data['text'])
            candidates = [phrase for phrase in candidates if phrase.split(' ')[0] not in stopwords]

            text_tokens = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'].squeeze(0))
            candidates_indices = {
                phrase: get_phrase_indices(text_tokens, phrase, prefix) 
                for phrase in candidates 
                if get_phrase_indices(text_tokens, phrase, prefix)
            }

            candidates_indices = remove_repeated_sub_word(candidates_indices)

            for layer in range(len(attentions)):
                for head in range(attentions[layer].size(1)):
                    attention_map = attentions[layer].squeeze(0)[head]
                    global_attention_scores = get_col_sum_token_level(attention_map)

                    # Adjust the first or last token's attention based on the model
                    if args.plm in ["GPT2", "RoBERTa", "XLM-R"]:
                        global_attention_scores[0] = 0   # Zero out attention for the <s> token at the beginning
                        global_attention_scores[-1] = 0  # Zero out attention for the </s> token at the end

                    redistributed_attention_map = redistribute_global_attention_score(attention_map, global_attention_scores)
                    redistributed_attention_map = normalize_attention_map(redistributed_attention_map)
                    proportional_attention_scores = get_row_sum_token_level(redistributed_attention_map)

                    if args.mode == 'Both':
                        final_tokens_score = global_attention_scores + proportional_attention_scores
                    elif args.mode == 'Global':
                        final_tokens_score = global_attention_scores
                    elif args.mode == 'Proportional':
                        final_tokens_score = proportional_attention_scores

                    phrase_score_dict = {
                        phrase: aggregate_phrase_scores(indices, final_tokens_score) / (len(indices) if len(phrase.split()) == 1 else 1)
                        for phrase, indices in candidates_indices.items() if indices
                    }

                    sorted_scores = sorted(phrase_score_dict.items(), key=lambda item: item[1], reverse=True)
                    stemmed_sorted_scores = [(" ".join(stemmer.stem(word) for word in phrase.split()), score) for phrase, score in sorted_scores]

                    set_stemmed_scores_list = list(dict.fromkeys([phrase for phrase, _ in stemmed_sorted_scores]))
                    pred_stemmed_phrases = set_stemmed_scores_list[:15]
                    layer_head_predicted_top15[(layer, head)].append(pred_stemmed_phrases)

    top3_f1_5, top3_f1_10, top3_f1_15 = evaluate_all_heads(layer_head_predicted_top15, dataset)

    # Output top results
    print("Top@5 F1 - Top 3 heads:")
    print(top3_f1_5[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("Top@10 F1 - Top 3 heads:")
    print(top3_f1_10[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("top@15_f1  Top3 heads:")
    print(top3_f1_15[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))


def rank_long_documents(args, dataset, model, tokenizer):
    # Define token prefixes and maximum lengths for different models
    model_settings = {
        'BERT': {'prefix': '##', 'max_len': 512},
        'mBERT': {'prefix': '##', 'max_len': 512},
        'GPT2': {'prefix': 'Ġ', 'max_len': 1024},
        'RoBERTa': {'prefix': 'Ġ', 'max_len': 1024}
    }

    # Get model specific settings
    prefix = model_settings.get(args.plm, {'prefix': '##', 'max_len': 512})

    layer_head_predicted_top15 = defaultdict(list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Using device: {device}")

    for data in tqdm(dataset, desc="Processing long documents"):
        with torch.no_grad():
            tokenized_text = tokenizer(data['text'], return_tensors='pt')
            candidates = [phrase for phrase in get_candidates(pos_tagger, data['text']) if phrase.split(' ')[0] not in stopwords]
            total_tokens_ids = tokenized_text['input_ids'].squeeze(0).tolist()

            # Adjust token ids based on the model
            if args.plm in ['BERT', 'mBERT']:
                total_tokens_ids = total_tokens_ids[1:-1]  # Remove CLS and SEP tokens for BERT and mBERT

            # Generate segments and attention masks
            windows, attention_masks = get_same_len_segments(total_tokens_ids, prefix['max_len'])

            layer_head_scores = defaultdict(lambda: defaultdict(float))

            for i, (window, attention_mask) in enumerate(zip(windows, attention_masks)):
                # Prepare input tensors
                window_tensor = torch.tensor([window]).to(device)
                attention_mask_tensor = torch.tensor([attention_mask]).to(device)

                outputs = model(window_tensor, attention_mask=attention_mask_tensor)
                attentions = outputs.attentions

                text_tokens = tokenizer.convert_ids_to_tokens(window_tensor[0])
                candidates_indices = {phrase: get_phrase_indices(text_tokens, phrase, prefix['prefix']) for phrase in candidates if get_phrase_indices(text_tokens, phrase, prefix['prefix'])}
                candidates_indices = remove_repeated_sub_word(candidates_indices)

                # Process each layer and head
                for layer in range(12):  # Assuming the model has 12 layers; adapt as necessary
                    for head in range(12):  # Assuming each layer has 12 heads; adapt as necessary
                        attention_map = attentions[layer][head].squeeze(0)
                        global_attention_scores = get_col_sum_token_level(attention_map)
                        if args.plm in ["GPT2", "RoBERTa"]:  # Zero out attention for special tokens
                            global_attention_scores[0], global_attention_scores[-1] = 0, 0

                        # Normalize and redistribute attention
                        redistributed_attention_map = normalize_attention_map(redistribute_global_attention_score(attention_map, global_attention_scores))
                        proportional_attention_scores = get_row_sum_token_level(redistributed_attention_map)

                        # Calculate final token scores based on the specified mode
                        final_tokens_score = {
                            'Both': global_attention_scores + proportional_attention_scores,
                            'Global': global_attention_scores,
                            'Proportional': proportional_attention_scores
                        }[args.mode]

                        # Calculate scores for each phrase
                        for phrase, indices in candidates_indices.items():
                            final_phrase_score = aggregate_phrase_scores(indices, final_tokens_score) / (len(indices) if len(phrase.split()) == 1 else 1)
                            layer_head_scores[(layer, head)][phrase] += final_phrase_score

            # Collate results for all heads
            for (layer, head), scores in layer_head_scores.items():
                sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                stemmed_scores = [(" ".join(stemmer.stem(word) for word in phrase.split()), score) for phrase, score in sorted_scores]
                unique_stemmed_scores = list(dict.fromkeys(stemmed_scores))
                layer_head_predicted_top15[(layer, head)].append([phrase for phrase, _ in unique_stemmed_scores][:15])

    top3_f1_5, top3_f1_10, top3_f1_15 = evaluate_all_heads(layer_head_predicted_top15, dataset)
    print("Top F1 scores for short documents:\n")
    print("Top@5 F1 - Top 3 heads:\n", top3_f1_5[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("Top@10 F1 - Top 3 heads:\n", top3_f1_10[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))
    print("Top@15 F1 - Top 3 heads:\n", top3_f1_15[['f1@5', 'f1@10', 'f1@15', 'layer', 'head']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default='lnspec',
                        type=str,
                        required=True,
                        help="Dataset name with language combinations")

    parser.add_argument("--plm",
                        default='BERT',
                        type=str,
                        required=True,
                        help="BERT or GPT2")
    parser.add_argument("--mode",
                        default='Both',
                        type=str,
                        required=False,
                        help="Both or Global or Proportional")

    args = parser.parse_args()

    language_combinations = {
        'En_Es_Inspec': ('English', 'Spanish'),
        'En_Fr_Inspec': ('English', 'French'),
        'En_It_Inspec': ('English', 'Italian'),
        'English_Inspec': ('English',),
        'Spanish_Inspec': ('Spanish',),
        'French_Inspec': ('French',),
        'Italian_Inspec': ('Italian',),
        'En_Es_SemEval2010': ('English', 'Spanish'),
        'En_Fr_SemEval2010': ('English', 'French'),
        'En_It_SemEval2010': ('English', 'Italian'),
        'En_Es_SemEval2017': ('English', 'Spanish'),
        'En_Fr_SemEval2017': ('English', 'French'),
        'En_It_SemEval2017': ('English', 'Italian'),
        'English_SemEval2010': ('English',),
        'Spanish_SemEval2010': ('Spanish',),
        'French_SemEval2010': ('French',),
        'Italian_SemEval2010': ('Italian',),
        'English_SemEval2017': ('English',),
        'Spanish_SemEval2017': ('Spanish',),
        'French_SemEval2017': ('French',),
        'Italian_SemEval2017': ('Italian',)
        # Add more language combinations as needed
    }

    if args.dataset not in language_combinations:
        print('Invalid dataset')
        sys.exit(1)

    languages = language_combinations[args.dataset]

    data_path = f'data/{args.dataset}.jsonl'
    dataset = read_jsonl(data_path)

    if args.plm == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, add_pooling_layer=False)
    elif args.plm == 'GPT2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2', output_hidden_states=True, output_attentions=True)
    elif args.plm == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base', output_attentions=True)
    elif args.plm == 'mBERT':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertModel.from_pretrained("bert-base-multilingual-cased", output_attentions=True, add_pooling_layer=False)

    # if 'English_Inspec' in args.dataset or 'English_SemEval2017' in args.dataset:
    doc_type = 'short'
    # else:
    #     doc_type = 'long'

    if doc_type == 'short':
        rank_short_documents(args, dataset, model, tokenizer)
    elif doc_type == 'long':
        rank_long_documents(args, dataset, model,tokenizer)
