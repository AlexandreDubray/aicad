# Python API reference

## Modelling

```{eval-rst}
.. currentmodule:: pyaicad

.. autoclass:: Model
   :members:

.. autoclass:: Var
   :members:

.. autoclass:: Constraint
   :members:

.. autofunction:: all_different
.. autofunction:: not_equal
.. autofunction:: equal
.. autofunction:: among
.. autofunction:: at_least
.. autofunction:: sum_equals
.. autofunction:: gcc
.. autofunction:: regular
```

## Compiling

```{eval-rst}
.. autoclass:: Compiler
   :members:

.. autoclass:: MDD
   :members:
```

## Local search

```{eval-rst}
.. autofunction:: local_search

.. autoclass:: BeliefPropagationSearch
   :members:

.. autoclass:: NeuralPriorSearch
   :members:

.. autoclass:: NeuralPriorBPSearch
   :members:

.. autoclass:: Limits
   :members:

.. autoclass:: Result
   :members:

.. autoclass:: Status
   :members:
```

## Settings and errors

```{eval-rst}
.. autofunction:: set_seed
.. autofunction:: set_num_threads
.. autofunction:: get_num_threads
.. autofunction:: set_log_level

.. autodata:: HAS_LEARNING

.. autoexception:: AicadError
.. autoexception:: ModelError
.. autoexception:: CheckpointError
.. autoexception:: FeatureUnavailableError
```

## Training

These need the full build.

```{eval-rst}
.. automodule:: pyaicad.learning
   :members:
```
