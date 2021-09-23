import numpy as np

from pywt._c99_config import _have_c99_complex
from pywt._extensions._dwt import dwt_axis
from pywt._utils import _wavelets_per_axis, _modes_per_axis
from pywt._multilevel import _check_level


def partial_dwptn(data, wavelet, mode='symmetric', axes=None, approx_only=False):
    """
    Partial Single-level n-dimensional Discrete Wavelet Packet Transform.
    Computes only approximation subband
    Parameters
    ----------
    data : array_like
        n-dimensional array with input data.
    wavelet : Wavelet object or name string, or tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or tuple of string, optional
        Signal extension mode used in the decomposition,
        see :ref:`Modes <ref-modes>`. This can also be a tuple of modes
        specifying the mode to use on each axis in ``axes``.
    axes : sequence of ints, optional
        Axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes. A value of ``None`` (the
        default) selects all axes.
        Axes may be repeated, but information about the original size may be
        lost if it is not divisible by ``2 ** nrepeats``. The reconstruction
        will be larger, with additional values derived according to the
        ``mode`` parameter. ``pywt.wavedecn`` should be used for multilevel
        decomposition.
    approx_only : bool, optional
        When set to true, only the approximation subband is returned.
    Returns
    -------
    coeffs : dict
        Results are arranged in a dictionary, where key specifies
        the transform type on each dimension and value is a n-dimensional
        coefficients array.
        For example, for a 2D case the result will look something like this::
            {'aa': <coeffs>  # A(LL) - approx. on 1st dim, approx. on 2nd dim
             'ad': <coeffs>  # V(LH) - approx. on 1st dim, det. on 2nd dim
             'da': <coeffs>  # H(HL) - det. on 1st dim, approx. on 2nd dim
             'dd': <coeffs>  # D(HH) - det. on 1st dim, det. on 2nd dim
            }
        For user-specified ``axes``, the order of the characters in the
        dictionary keys map to the specified ``axes``.
    """
    data = np.asarray(data)
    if not _have_c99_complex and np.iscomplexobj(data):
        real = partial_dwptn(data.real, wavelet, mode, axes)
        imag = partial_dwptn(data.imag, wavelet, mode, axes)
        return dict((k, real[k] + 1j * imag[k]) for k in real.keys())

    if data.dtype == np.dtype('object'):
        raise TypeError("Input must be a numeric array-like")
    if data.ndim < 1:
        raise ValueError("Input data must be at least 1D")

    if axes is None:
        axes = range(data.ndim)
    axes = [a + data.ndim if a < 0 else a for a in axes]

    modes = _modes_per_axis(mode, axes)
    wavelets = _wavelets_per_axis(wavelet, axes)

    coeffs = [('', data)]
    for axis, wav, mode in zip(axes, wavelets, modes):
        new_coeffs = []
        for subband, x in coeffs:
            cA, cD = dwt_axis(x, wav, mode, axis)
            if approx_only:
                new_coeffs.append((subband + 'a', cA))
            else:
                new_coeffs.extend([(subband + 'a', cA),
                                   (subband + 'd', cD)])
        coeffs = new_coeffs
    return dict(coeffs)


def partial_dwpt2(data, wavelet, mode='symmetric', axes=(-2, -1), approx_only=False):
    """
    Partial 2D Discrete Wavelet Packet Transform.
    Parameters
    ----------
    data : array_like
        2D array with input data
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or 2-tuple of strings, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple of modes specifying the mode to use on each axis in
        ``axes``.
    axes : 2-tuple of ints, optional
        Axes over which to compute the DWT. Repeated elements mean the DWT will
        be performed multiple times along these axes.
    approx_only : bool, optional
        When set to true, only the approximation subband is returned.
    Returns
    -------
    (cA, (cH, cV, cD)) : tuple
        Approximation, horizontal detail, vertical detail and diagonal
        detail coefficients respectively.  Horizontal refers to array axis 0
        (or ``axes[0]`` for user-specified ``axes``).
    Examples
    --------
    >>> import numpy as np
    >>> import pywt
    >>> data = np.ones((4,4), dtype=np.float64)
    >>> coeffs = pywt.dwt2(data, 'haar')
    >>> cA, (cH, cV, cD) = coeffs
    >>> cA
    array([[ 2.,  2.],
           [ 2.,  2.]])
    >>> cV
    array([[ 0.,  0.],
           [ 0.,  0.]])
    """
    axes = tuple(axes)
    data = np.asarray(data)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")
    if data.ndim < len(np.unique(axes)):
        raise ValueError("Input array has fewer dimensions than the specified "
                         "axes")

    coefs = partial_dwptn(data, wavelet, mode, axes, approx_only)

    if approx_only:
        return coefs['aa']
    else:
        return coefs['aa'], (coefs['da'], coefs['ad'], coefs['dd'])


def partial_waveletpacketdec2(data, wavelet, mode='symmetric', level=None, axes=(-2, -1)):
    """
    Multilevel 2D Partial Discrete Wavelet Packet Transform.
    Parameters
    ----------
    data : ndarray
        2D input data
    wavelet : Wavelet object or name string, or 2-tuple of wavelets
        Wavelet to use.  This can also be a tuple containing a wavelet to
        apply along each axis in ``axes``.
    mode : str or 2-tuple of str, optional
        Signal extension mode, see :ref:`Modes <ref-modes>`. This can
        also be a tuple containing a mode to apply along each axis in ``axes``.
    level : int, optional
        Decomposition level (must be >= 0). If level is None (default) then it
        will be calculated using the ``dwt_max_level`` function.
    axes : 2-tuple of ints, optional
        Axes over which to compute the DWT. Repeated elements are not allowed.
    Returns
    -------
    [cAn, (cEHn, cEVn, cEDn), ... (cEH1, cEV1, cED1)] : list
        Coefficients list.  For user-specified ``axes``, ``cEH*``
        corresponds to ``axes[0]`` while ``cEV*`` corresponds to ``axes[1]``.
        The first element returned is the approximation coefficients for the
        nth level of decomposition.  Remaining elements are tuples of edge
        coefficients in descending order of decomposition level, as proposed in
        "A novel discrete wavelet transform framework for full reference image quality assessment" S. Rezazadeh, S. Coulombe
        cEHl = H_{l}A_{n-l}, cEVl = V_{l}A_{n-l}, cEDl = D_{l}A_{n-l} in the paper above.

    TODO : Update example
    Examples
    --------
    >>> import pywt
    >>> import numpy as np
    >>> coeffs = pywt.wavedec2(np.ones((4,4)), 'db1')
    >>> # Levels:
    >>> len(coeffs)-1
    2
    >>> pywt.waverec2(coeffs, 'db1')
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])
    """
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("Expected input data to have at least 2 dimensions.")

    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("Expected 2 axes")
    if len(axes) != len(set(axes)):
        raise ValueError("The axes passed to wavedec2 must be unique.")
    try:
        axes_sizes = [data.shape[ax] for ax in axes]
    except IndexError:
        raise ValueError("Axis greater than data dimensions")

    wavelets = _wavelets_per_axis(wavelet, axes)
    dec_lengths = [w.dec_len for w in wavelets]

    level = _check_level(axes_sizes, dec_lengths, level)

    temp_coeffs_list = []
    a = data
    for i in range(level):
        a, ds = partial_dwpt2(a, wavelet, mode, axes, approx_only=False)
        temp_coeffs_list.append(ds)

    coeffs_list = []
    for i, temp_coeffs_level in enumerate(temp_coeffs_list):
        coeffs_level = []
        for subband, coeff in enumerate(temp_coeffs_level):
            for _ in range(level-1-i):
                coeff = partial_dwpt2(coeff, wavelet, mode, axes, approx_only=True)
            coeffs_level.append(coeff)
        coeffs_list.append(tuple(coeffs_level))
    coeffs_list.append(a)

    coeffs_list.reverse()
    return coeffs_list
