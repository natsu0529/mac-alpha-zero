import mlx.core as mx


def test_mlx_runs_on_cpu():
    # CI runners may lack a Metal GPU, so check the CPU path explicitly.
    with mx.stream(mx.cpu):
        x = mx.arange(4) * 2
        mx.eval(x)
    assert x.tolist() == [0, 2, 4, 6]
