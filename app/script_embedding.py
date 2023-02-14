# %%
import networkx as nx

import numpy as np
import torch
from tqdm import tqdm

from rostok.graph_grammar.node import GraphGrammar
from rostok.graph_grammar.node_vocabulary import NodeVocabulary
from rostok.graph_grammar.rule_vocabulary import RuleVocabulary
import rule_without_chrono as re

# %%

def get_input_layer(node, dict_id_label_nodes):
    input = torch.zeros(len(dict_id_label_nodes)).long()
    input[node] = 1
    return input

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

def random_batch(skip_grams):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), 2, replace=False)

    for i in random_index:
        random_inputs.append(skip_grams[i][0])  # target
        random_labels.append(skip_grams[i][1])  # context word

    return random_inputs, random_labels

# %%

class skipgramm_model(torch.nn.Module):

    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.W = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.WT = torch.nn.Linear(embedding_size, vocabulary_size, bias=False)

    def forward(self, x):
        embdedings = self.embedding(x)
        hidden_layer = torch.nn.functional.relu(self.W(embdedings))
        output_layer = self.WT(hidden_layer)

        return output_layer

    def get_node_embedding(self, node, sorted_node_labels, dict_label_id_nodes):
        input = torch.zeros(len(sorted_node_labels)).float()
        input[dict_label_id_nodes[node]] = 1
        return self.embedding(input).view(1, -1)


def skipgram(paths, dict_label_id_nodes, window_size=1):
    idx_pairs = []
    for path in paths:
        indices = [dict_label_id_nodes[node_label] for node_label in path]
        for pos_center_node, node_index in enumerate(indices):
            for i in range(-window_size, window_size + 1):
                pos_context_node = pos_center_node + i

                if pos_context_node < 0 or pos_context_node >= len(
                        indices) or pos_center_node == pos_context_node:
                    continue
                context_id_node = indices[pos_context_node]
                idx_pairs.append((node_index, context_id_node))

    return np.array(idx_pairs)


def create_dict_node_labels(node_vocabulary: NodeVocabulary):

    sorted_node_labels = sorted(node_vocabulary.node_dict.keys())

    dict_id_label_nodes = dict(enumerate(sorted_node_labels))
    dict_label_id_nodes = {w: idx for (idx, w) in enumerate(sorted_node_labels)}

    return dict_id_label_nodes, dict_label_id_nodes

# %%
rule_vocab = re.init_extension_rules()
node_vocabulary = rule_vocab.node_vocab

id2label, label2id = create_dict_node_labels(node_vocabulary)

model = skipgramm_model(len(id2label), 2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

graph = vocabulary2batch_graph(rule_vocab, 15)
pairs = skipgram(graph.get_uniq_representation(),label2id)
for epoch in tqdm(range(150000), total=len(pairs)):
    input_batch, target_batch = random_batch(pairs)
    input_batch = get_input_layer(input_batch, id2label)
    target_batch = get_input_layer(target_batch, id2label)

    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, voc_size], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 10000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), ' cost =', '{:.6f}'.format(loss))

    loss.backward(retain_graph=True)
    optimizer.step()


