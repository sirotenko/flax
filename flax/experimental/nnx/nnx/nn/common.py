# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utilities for nnx modules."""

import typing as tp

import jax.numpy as jnp
from jax.core import Shape
from jax.typing import DTypeLike


@tp.runtime_checkable
class HasCacheInitializer(tp.Protocol):
  def init_cache(
    self,
    input_shape: Shape,
    dtype: DTypeLike = jnp.float32,
  ):
    ...
