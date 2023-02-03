import time
from rostok.graph_grammar.node import GraphGrammar
from example_vocabulary import rule_vocab
from rostok.graph_grammar.graphgrammar_explorer import ruleset_explorer

start = time.time()
out = ruleset_explorer(3, rule_vocab)
for i in out[0]:
    print(i.nodes(True))
ex = time.time() - start

print(f"time :{ex}")
print(f"Non-uniq graphs :{out[1]}")
print(f"Uniq graphs :{len(out[0])}")
