# Initial real Phase0 attempt — 2026-11-01

Source362e8a6; SCR/PM-small both failed at the first contraction.
No measurements were accepted and no gate was evaluated. Config/seed unchanged.
T8/T10 were rerun after fixing dtype preservation:4 passed in5.22s.

## scr

```text
  File "/Users/pxokq/expanding-guide model/ett/phase0/measure_ge.py", line 86, in paired
    records=engine.contract(eval_batches[0],loss_fn,acceptance=False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/ett/core/ett.py", line 98, in contract
    compact=measured_loss(trial_model,check_batch,loss_fn);compact_eval=measured_loss(trial_model,check_batch,loss_fn,False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/ett/core/ett.py", line 16, in measured_loss
    with measurement(model,training),torch.no_grad():return float(loss_fn(model(batch[0]),batch[1]))
                                                                          ^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: self and mat2 must have the same dtype, but got Float and Double
```

## pmnist

```text
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/ett/core/ett.py", line 98, in contract
    compact=measured_loss(trial_model,check_batch,loss_fn);compact_eval=measured_loss(trial_model,check_batch,loss_fn,False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/ett/core/ett.py", line 16, in measured_loss
    with measurement(model,training),torch.no_grad():return float(loss_fn(model(batch[0]),batch[1]))
                                                                          ^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/pxokq/expanding-guide model/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: self and mat2 must have the same dtype, but got Float and Double
Extracting /tmp/ett_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to /tmp/ett_data/MNIST/raw

```

