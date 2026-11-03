"""Meaningful equality checks; run only after the numerical timing lease."""
from io import StringIO
import json

import numpy as np

from scripts.build_expanded_policy_packages import TextCatalog, TrieIndex
from scripts.public_qa_benchmark import Index, graph, local_view, seed_vector


def test_trie_keeps_all_prefix_alias_duplicate_and_parenthesis_ids():
    docs = [('New York', 'x'), ('New York City', 'y'), ('NEW YORK (state)', 'z'),
        ('New York', 'different text'), ('York', 'w'), ('(empty)', 'unused'),
        ('A', 'single-letter title'), ('C++', 'punctuation-normalized title')]
    reference, accelerated = Index(docs), TrieIndex(docs)
    text = 'A New York City event in NEW YORK; C++ is present.'
    expected = {0, 1, 2, 3, 4, 6, 7}
    assert reference.mentions(text) == accelerated.mentions(text) == expected
    assert accelerated.mentions('New Yorkshire') == reference.mentions('New Yorkshire') == set()
    assert accelerated.mentions('empty') == reference.mentions('empty') == set()


def test_trie_randomized_mentions_bm25_complete_graph_and_source_policy_equal():
    rng = np.random.default_rng(7301)
    vocabulary = ['east', 'west', 'city', 'river', 'new', 'old', 'station', 'park']
    docs = []
    for i in range(40):
        title = ' '.join(rng.choice(vocabulary, size=int(rng.integers(1, 5))))
        if i % 5 == 0:
            title += ' (district)'
        body = ' '.join(rng.choice(vocabulary, size=30))
        docs.append((title, body))
    reference, accelerated = Index(docs), TrieIndex(docs)
    texts = [body for _, body in docs]+[title for title, _ in docs]
    texts += [' '.join(rng.choice(vocabulary, size=50)) for _ in range(50)]
    for text in texts:
        assert accelerated.mentions(text) == reference.mentions(text)
        np.testing.assert_array_equal(accelerated.bm25(text), reference.bm25(text))
    P, entities = graph(reference)
    Q, accelerated_entities = graph(accelerated)
    np.testing.assert_array_equal(P.indptr, Q.indptr)
    np.testing.assert_array_equal(P.indices, Q.indices)
    np.testing.assert_array_equal(P.data, Q.data)
    assert entities == accelerated_entities
    for question in texts[:10]:
        a, b = reference.bm25(question), accelerated.bm25(question)
        np.testing.assert_array_equal(seed_vector(a), seed_vector(b))
        s, D = local_view(P, entities, reference, question, a)
        t, E = local_view(Q, accelerated_entities, accelerated, question, b)
        np.testing.assert_array_equal(s, t)
        np.testing.assert_array_equal(D, E)


def test_dev_first_catalog_preserves_conflicts_and_local_duplicate_occurrences():
    docs, mappings = StringIO(), StringIO()
    catalog = TextCatalog(docs, mappings)
    assert catalog.add_record([(0, 'Same', 'one'), (1, 'Same', 'one'),
        (2, 'Same', 'two')], 'dev', 0, 'devqid') == [0, 1, 2]
    catalog.dev_nodes = len(catalog.keys)
    assert catalog.add_record([(4, 'Same', 'one'), (5, 'Same', 'one'),
        (6, 'Same', 'two'), (7, 'New', 'train text')], 'train', 0) == [0, 1, 2, 3]
    records = [json.loads(line) for line in docs.getvalue().splitlines()]
    assert [(r['title'], r['text'], r['occurrence_ordinal']) for r in records] == [
        ('Same', 'one', 0), ('Same', 'one', 1), ('Same', 'two', 0), ('New', 'train text', 0)]
    train_map = [json.loads(line) for line in mappings.getvalue().splitlines()
        if json.loads(line)['split'] == 'train']
    assert all('qid' not in row for row in train_map)
    assert catalog.counts['train_occurrences_reusing_dev_nodes'] == 3
