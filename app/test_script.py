# %%
import random
import re
from dataclasses import dataclass
from time import monotonic
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader
from tqdm import tqdm

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
import rule_without_chrono as re


# %%
def vocabulary2batch_graph(rule_vocabulary: RuleVocabulary, max_rules: int):

    batch_graph = GraphGrammar()
    amount_rules = np.random.randint(1, max_rules)
    for _ in range(amount_rules):
        rules = rule_vocabulary.get_list_of_applicable_rules(batch_graph)
        if len(rules) > 0:
            rule = rule_vocabulary.get_rule(rules[np.random.choice(len(rules))])
            batch_graph.apply_rule(rule)
        else:
            break
    return batch_graph


def create_train_valid_data(rule_vocabulary: RuleVocabulary, amount_graph: int,
                            pseudo_length_graph: int):
    train_data = []
    for __ in range(round(amount_graph * 0.8)):
        flatted_graph = []
        graph = vocabulary2batch_graph(rule_vocabulary, pseudo_length_graph)
        df_travels = graph.get_uniq_representation()
        for path in df_travels:
            flatted_graph = flatted_graph + path
        train_data.append(flatted_graph)
    valid_data = []
    for __ in range(round(amount_graph * 0.2)):
        flatted_graph = []
        graph = vocabulary2batch_graph(rule_vocabulary, pseudo_length_graph)
        df_travels = graph.get_uniq_representation()
        for path in df_travels:
            flatted_graph = flatted_graph + path
        valid_data.append(flatted_graph)
    return train_data, valid_data


# %%
@dataclass
class Word2VecParams:

    # skipgram parameters
    MIN_FREQ = 1
    SKIPGRAM_N_WORDS = 1
    T = 85
    NEG_SAMPLES = 50
    NS_ARRAY_LEN = 200

    # network parameters
    BATCH_SIZE = 128
    EMBED_DIM = 10
    EMBED_MAX_NORM = None
    N_EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CRITERION = nn.BCEWithLogitsLoss()


# %%
class model_vocabulary:

    def __init__(self, node_vocabulary: NodeVocabulary):

        sorted_node_labels = sorted(node_vocabulary.node_dict.keys())

        self.itos = dict(enumerate(sorted_node_labels))
        self.stoi = {w: idx for (idx, w) in enumerate(sorted_node_labels)}

    def __len__(self):
        return len(self.stoi) - 1

    def get_index(self, label_node: Union[str, List]):

        if isinstance(label_node, str):
            if label_node in self.stoi:
                return self.stoi.get(label_node)

        elif isinstance(label_node, list):
            res = []
            for n in label_node:
                if n in self.stoi:
                    res.append(self.stoi.get(n))
            return res
        else:
            raise ValueError(f"Label node {label_node} is not a string or a list of strings.")

    def lookup_token(self, token: Union[int, List]):
        if isinstance(token, (int, np.int64)):
            if token in self.itos:
                return self.itos.get(token)
            else:
                raise ValueError(f"Token {token} not in vocabulary")
        elif isinstance(token, list):
            res = []
            for t in token:
                if t in self.itos:
                    res.append(self.itos.get(token))
                else:
                    raise ValueError(f"Token {t} is not a valid index.")
            return res


# %%
def calculate_frequency_nodes(vocab: model_vocabulary, flatted_graphs: list):
    frequency_nodes = {label: 0 for label in vocab.stoi.keys()}

    for graph in flatted_graphs:
        for node in graph:
            frequency_nodes[node] = int(frequency_nodes.get(node, 0) + 1)
    total_nodes = np.nansum([f for f in frequency_nodes.values()], dtype=int)

    return frequency_nodes, total_nodes


# %%
class SkipGrams:

    def __init__(self, vocab: model_vocabulary, flatted_graph: list, params: Word2VecParams):
        self.vocab = vocab
        self.params = params

        freq_dict, total_tokens = calculate_frequency_nodes(self.vocab, flatted_graph)

        self.t = self._t(freq_dict, total_tokens)
        self.discard_probs = self._create_discard_dict(freq_dict, total_tokens)

    def _t(self, freq_dict, total_tokens):
        freq_list = []
        for freq in list(freq_dict.values())[1:]:
            freq_list.append(freq / total_tokens)
        return np.percentile(freq_list, self.params.T)

    def _create_discard_dict(self, freq_dict, total_tokens):
        discard_dict = {}
        for node, freq in freq_dict.items():
            dicard_prob = 1 - np.sqrt(self.t / (freq / total_tokens + self.t))
            discard_dict[self.vocab.stoi[node]] = dicard_prob
        return discard_dict

    def collate_skipgram(self, batch):
        batch_input, batch_output = [], []
        for graph in batch:
            node_tokens = self.vocab.get_index(graph)

            if len(node_tokens) < self.params.SKIPGRAM_N_WORDS * 2 + 1:
                continue

            for idx in range(len(node_tokens) - self.params.SKIPGRAM_N_WORDS * 2):
                token_id_sequence = node_tokens[idx:(idx + self.params.SKIPGRAM_N_WORDS * 2 + 1)]
                input_ = token_id_sequence.pop(self.params.SKIPGRAM_N_WORDS)
                outputs = token_id_sequence

                prb = random.random()
                del_pair = self.discard_probs.get(input_)
                if input_ == 0 or del_pair >= prb:
                    continue
                else:
                    for output in outputs:
                        prb = random.random()
                        del_pair = self.discard_probs.get(output)
                        if output == 0 or del_pair >= prb:
                            continue
                        else:
                            batch_input.append(input_)
                            batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)

        return batch_input, batch_output


# %%
class NegativeSampler:

    def __init__(self, vocab: model_vocabulary, train_graphs: list, ns_exponent: float,
                 ns_array_len: int):
        self.vocab = vocab
        self.ns_exponent = ns_exponent
        self.ns_array_len = ns_array_len
        self.ns_array = self._create_negative_sampling(train_graphs)

    def __len__(self):
        return len(self.ns_array)

    def _create_negative_sampling(self, train_graphs: list):

        frequency_dict, total_tokens = calculate_frequency_nodes(self.vocab, train_graphs)
        frequency_dict_scaled = {
            self.vocab.stoi[node]: max(1, int((freq / total_tokens) * self.ns_array_len))
            for node, freq in frequency_dict.items()
        }
        ns_array = []
        for node, freq in tqdm(frequency_dict_scaled.items()):
            ns_array = ns_array + [node] * freq
        return ns_array

    def sample(self, n_batches: int = 1, n_samples: int = 1):
        samples = []
        for _ in range(n_batches):
            samples.append(random.sample(self.ns_array, n_samples))
        samples = torch.as_tensor(np.array(samples))
        return samples


# %%
# Model
class Model(nn.Module):

    def __init__(self, vocab: model_vocabulary, params: Word2VecParams):
        super().__init__()
        self.vocab = vocab
        self.t_embeddings = nn.Embedding(self.vocab.__len__() + 1,
                                         params.EMBED_DIM,
                                         max_norm=params.EMBED_MAX_NORM)
        self.c_embeddings = nn.Embedding(self.vocab.__len__() + 1,
                                         params.EMBED_DIM,
                                         max_norm=params.EMBED_MAX_NORM)

    def forward(self, inputs, context):
        # getting embeddings for target & reshaping
        target_embeddings = self.t_embeddings(inputs)
        n_examples = target_embeddings.shape[0]
        n_dimensions = target_embeddings.shape[1]
        target_embeddings = target_embeddings.view(n_examples, 1, n_dimensions)

        # get embeddings for context labels & reshaping
        # Allows us to do a bunch of matrix multiplications
        context_embeddings = self.c_embeddings(context)
        # * This transposes each batch
        context_embeddings = context_embeddings.permute(0, 2, 1)

        # * custom linear layer
        dots = target_embeddings.bmm(context_embeddings)
        dots = dots.view(dots.shape[0], dots.shape[2])
        return dots

    def normalize_embeddings(self):
        embeddings = list(self.t_embeddings.parameters())[0]
        embeddings = embeddings.cpu().detach().numpy()
        norms = (embeddings**2).sum(axis=1)**(1 / 2)
        norms = norms.reshape(norms.shape[0], 1)
        return embeddings / norms

    def get_similar_node(self, node, n):
        node_id = self.vocab.get_index(node)

        embedding_norms = self.normalize_embeddings()
        node_vec = embedding_norms[node_id]
        node_vec = np.reshape(node_vec, (node_vec.shape[0], 1))
        dists = np.matmul(embedding_norms, node_vec).flatten()
        topN_ids = np.argsort(-dists)[1:n + 1]

        topN_dict = {}
        for sim_node_id in topN_ids:
            sim_node = self.vocab.lookup_token(sim_node_id)
            topN_dict[sim_node] = dists[sim_node_id]
        return topN_dict

    def get_similarity(self, node_1, node_2):
        idx1 = self.vocab.get_index(node_1)
        idx2 = self.vocab.get_index(node_2)
        if idx1 == 0 or idx2 == 0:
            print("One or both words are out of vocabulary")
            return

        embedding_norms = self.normalize_embeddings()
        node1_vec, node2_vec = embedding_norms[idx1], embedding_norms[idx2]

        return cosine(node1_vec, node2_vec)


# %%
class Trainer:

    def __init__(self, model: Model, params: Word2VecParams, optimizer, vocab: model_vocabulary,
                 train_iter, valid_iter, skipgrams: SkipGrams):
        self.model = model
        self.optimizer = optimizer
        self.vocab = vocab
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.skipgrams = skipgrams
        self.params = params

        self.epoch_train_mins = {}
        self.loss = {"train": [], "valid": []}

        # sending all to device
        self.model.to(self.params.DEVICE)
        self.params.CRITERION.to(self.params.DEVICE)

        self.negative_sampler = NegativeSampler(vocab=self.vocab,
                                                ns_exponent=.75,
                                                train_graphs=self.train_iter,
                                                ns_array_len=self.params.NS_ARRAY_LEN)
        self.testnode = ['F1', 'J', 'L1', 'EM']

    def train(self):
        self.test_testnode()
        for epoch in range(self.params.N_EPOCHS):
            # Generate Dataloaders
            self.train_dataloader = DataLoader(self.train_iter,
                                               batch_size=self.params.BATCH_SIZE,
                                               shuffle=False,
                                               collate_fn=self.skipgrams.collate_skipgram)
            self.valid_dataloader = DataLoader(self.valid_iter,
                                               batch_size=self.params.BATCH_SIZE,
                                               shuffle=False,
                                               collate_fn=self.skipgrams.collate_skipgram)
            # training the model
            st_time = monotonic()
            self._train_epoch()
            self.epoch_train_mins[epoch] = round((monotonic() - st_time) / 60, 1)

            # validating the model
            self._validate_epoch()
            print(f"""Epoch: {epoch+1}/{self.params.N_EPOCHS}\n""",
                  f"""    Train Loss: {self.loss['train'][-1]:.2}\n""",
                  f"""    Valid Loss: {self.loss['valid'][-1]:.2}\n""",
                  f"""    Training Time (mins): {self.epoch_train_mins.get(epoch)}"""
                  """\n""")
            self.test_testnode()

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            if len(batch_data[0]) == 0:
                continue
            inputs = batch_data[0].to(self.params.DEVICE)
            pos_labels = batch_data[1].to(self.params.DEVICE)
            neg_labels = self.negative_sampler.sample(pos_labels.shape[0], self.params.NEG_SAMPLES)
            neg_labels = neg_labels.to(self.params.DEVICE)
            context = torch.cat([pos_labels.view(pos_labels.shape[0], 1), neg_labels], dim=1)

            # building the targets tensor
            y_pos = torch.ones((pos_labels.shape[0], 1))
            y_neg = torch.zeros((neg_labels.shape[0], neg_labels.shape[1]))
            y = torch.cat([y_pos, y_neg], dim=1).to(self.params.DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, context)
            loss = self.params.CRITERION(outputs, y)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)

        self.loss['train'].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.valid_dataloader, 1):
                if len(batch_data[0]) == 0:
                    continue
                inputs = batch_data[0].to(self.params.DEVICE)
                pos_labels = batch_data[1].to(self.params.DEVICE)
                neg_labels = self.negative_sampler.sample(
                    pos_labels.shape[0], self.params.NEG_SAMPLES).to(self.params.DEVICE)
                context = torch.cat([pos_labels.view(pos_labels.shape[0], 1), neg_labels], dim=1)

                # building the targets tensor
                y_pos = torch.ones((pos_labels.shape[0], 1))
                y_neg = torch.zeros((neg_labels.shape[0], neg_labels.shape[1]))
                y = torch.cat([y_pos, y_neg], dim=1).to(self.params.DEVICE)

                preds = self.model(inputs, context).to(self.params.DEVICE)
                loss = self.params.CRITERION(preds, y)

                running_loss.append(loss.item())

            epoch_loss = np.mean(running_loss)
            self.loss['valid'].append(epoch_loss)

    def test_testnode(self, n: int = 10):
        for node in self.testnode:
            print(node)
            nn_node = self.model.get_similar_node(node, n)
            for v, sim in nn_node.items():
                print(f"{v} ({sim:.3})", end=' ')
            print('\n')


# %%
rule_vocab = re.init_extension_rules()

# %%
params = Word2VecParams()
train_data, valid_data = create_train_valid_data(rule_vocab, 100000, 20)
vocab = model_vocabulary(rule_vocab.node_vocab)
skip_gram = SkipGrams(vocab=vocab, flatted_graph=train_data, params=params)
model = Model(vocab=vocab, params=params).to(params.DEVICE)
optimizer = torch.optim.Adam(params=model.parameters())

# %%
trainer = Trainer(model=model,
                  params=params,
                  optimizer=optimizer,
                  train_iter=train_data,
                  valid_iter=valid_data,
                  vocab=vocab,
                  skipgrams=skip_gram)
trainer.train()
None