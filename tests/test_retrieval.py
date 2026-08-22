from hello_agent.retrieval import VectorStore


def test_vector_store_returns_most_similar_text():
    store = VectorStore()

    store.add(text="The cat sat on the mat.", vector=[1.0, 0.0, 0.0])
    store.add(text="Dogs are loyal animals.", vector=[0.0, 1.0, 0.0])
    store.add(text="Cats are independent pets.", vector=[0.9, 0.1, 0.0])

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0][0] == "The cat sat on the mat."
    assert results[1][0] == "Cats are independent pets."


def test_vector_store_empty_returns_no_results():
    store = VectorStore()

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=3)

    assert results == []


def test_vector_store_top_k_limits_results():
    store = VectorStore()

    store.add(text="a", vector=[1.0, 0.0])
    store.add(text="b", vector=[0.9, 0.1])
    store.add(text="c", vector=[0.5, 0.5])

    results = store.search(query_vector=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0][0] == "a"
