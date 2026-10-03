# Development staging tree

The code as it was written and tested before it was ported into the TensorFold family (`patches/0002`):
`reference/` is the pure-PyTorch reference model and checkpoint loader used as the correctness oracle, `kernels/`
and `serving/` the first versions of the kernels and the serving layer (`scripts/campaign/port_serving.py` copied the
latter into the engine). The tests are in `../tests/`. The engine that runs is the patched TensorFold, not this tree.
