import jax.numpy as _jnp
import numpy as _np
from geomstats._backend._dtype_utils import _pre_set_default_dtype as set_default_dtype

# Mapping of string dtype representations to JAX dtypes
MAP_DTYPE = {
    "float32": _jnp.float32,
    "float64": _jnp.float64,
    "complex64": _jnp.complex64,
    "complex128": _jnp.complex128
}


def as_dtype(value):
    """
    Transform string representing dtype into JAX dtype.

    Parameters
    ----------
    value : str
        String representing the dtype to be converted.

    Returns
    -------
    dtype : jnp.dtype
        JAX dtype object corresponding to the input string.
    """
    return MAP_DTYPE[value]


def to_ndarray(x, dtype=None):
    """
    Convert a JAX array to a NumPy array, ensuring that any pending operations are completed.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The JAX array to convert.
    dtype : data-type, optional
        The desired data type for the NumPy array. If None, the dtype of the JAX array is used.

    Returns
    -------
    ndarray : numpy.ndarray
        A NumPy array representation of the input JAX array.
    """
    # First, ensure all JAX operations are completed
    x = x.block_until_ready()
    
    # If a dtype is specified and it's different from the current dtype, cast the array
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    
    # Convert to NumPy array
    return _np.array(x)