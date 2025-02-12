import numpy as np
from pynisher import limit, TimeoutException, MemoryLimitException
from functools import wraps
from time import time


# from collections.abc import Callable
# from typing import Union, Optional

def limited_execution(
  wrapper_function,
  task,
  *args,
  wall_time = None,
  cpu_time = None,
  memory = None,
  **kwargs,
  ):

    execution_assistant = limit(
      wrapper_function,
      memory=memory,
      wall_time=wall_time,
      cpu_time=cpu_time,
    )

    try:
        pred = execution_assistant(task, *args, **kwargs)
    except (TimeoutException, MemoryLimitException):
        pred = np.nan

    return pred


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        return result, te-ts
    return wrap
