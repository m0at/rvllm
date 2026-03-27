def test_import():
    import rvllm
    assert hasattr(rvllm, 'Sampler')
    assert hasattr(rvllm, 'Tokenizer')
    assert rvllm.__version__ == "0.1.0"
