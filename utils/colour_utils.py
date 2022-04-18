import numpy as np

__all__ = [
    'YCBCR_WEIGHTS', 'YCbCr_ranges', 'RGB_to_YCbCr', 'YCbCr_to_RGB',
    'RGB_to_YcCbcCrc', 'YcCbcCrc_to_RGB'
]


YCBCR_WEIGHTS = dict({
    'ITU-R BT.601': np.array([0.2990, 0.1140]),
    'ITU-R BT.709': np.array([0.2126, 0.0722]),
    'ITU-R BT.2020': np.array([0.2627, 0.0593]),
    'SMPTE-240M': np.array([0.2122, 0.0865])
})

WEIGHTS_YCBCR = dict(
    {
        "ITU-R BT.601": np.array([0.2990, 0.1140]),
        "ITU-R BT.709": np.array([0.2126, 0.0722]),
        "ITU-R BT.2020": np.array([0.2627, 0.0593]),
        "SMPTE-240M": np.array([0.2122, 0.0865]),
    }
)

def spow(a,p):
    sp = np.sign(a) * np.abs(a) ** p
    return sp

def eotf_ST2084(N,L_p=10000):
    """
    Define *SMPTE ST 2084:2014* optimised perceptual electro-optical transfer
    function (EOTF).

    This perceptual quantizer (PQ) has been modeled by Dolby Laboratories
    using *Barten (1999)* contrast sensitivity function.

    Parameters
    ----------
    N
        Color value abbreviated as :math:`N`, that is directly proportional to
        the encoded signal representation, and which is not directly

        proportional to the optical output of a display device.
    L_p
        System peak luminance :math:`cd/m^2`, this parameter should stay at its
        default :math:`10000 cd/m^2` value for practical applications. It is
        exposed so that the definition can be used as a fitting function.
    constants
        *SMPTE ST 2084:2014* constants.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
          Target optical output :math:`C` in :math:`cd/m^2` of the ideal
          reference display.

    Warnings
    --------
    *SMPTE ST 2084:2014* is an absolute transfer function.

    Notes
    -----
    -   *SMPTE ST 2084:2014* is an absolute transfer function, thus the
        domain and range values for the *Reference* and *1* scales are only
        indicative that the data is not affected by scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``N``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``C``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Miller2014a`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers2014a`

    Examples
    --------
    >>> eotf_ST2084(0.508078421517399)  # doctest: +ELLIPSIS
    100.0000000...
    """
    

    N = N.astype(np.float32)

    m_1=2610 / 4096 * (1 / 4)
    m_2=2523 / 4096 * 128
    c_1=3424 / 4096
    c_2=2413 / 4096 * 32
    c_3=2392 / 4096 * 32


    m_1_d = 1 / m_1
    m_2_d = 1 / m_2

    V_p = spow(N, m_2_d)
    n = np.maximum(0, V_p - c_1)
    L = spow((n / (c_2 - c_3 * V_p)), m_1_d)
    C = L_p * L

    return C.astype(np.float32)

def CV_range(bit_depth=10, is_legal=False, is_int=False):
    """
    Returns the code value :math:`CV` range for given bit depth, range legality
    and representation.

    Parameters
    ----------
    bit_depth : int, optional
        Bit depth of the code value :math:`CV` range.
    is_legal : bool, optional
        Whether the code value :math:`CV` range is legal.
    is_int : bool, optional
        Whether the code value :math:`CV` range represents integer code values.

    Returns
    -------
    ndarray
        Code value :math:`CV` range.

    Examples
    --------
    >>> CV_range(8, True, True)
    array([ 16, 235])
    >>> CV_range(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...])
    >>> CV_range(10, False, False)
    array([ 0.,  1.])
    """

    if is_legal:
        ranges = np.array([16, 235])
        ranges *= 2 ** (bit_depth - 8)
    else:
        ranges = np.array([0, 2 ** bit_depth - 1])

    if not is_int:
        ranges = ranges.astype(np.float32) / (2 ** bit_depth - 1)

    return ranges


def YCbCr_ranges(bits, is_legal, is_int):
    """"
    Returns the *Y'CbCr* colour encoding ranges array for given bit depth,
    range legality and representation.

    Parameters
    ----------
    bits : int
        Bit depth of the *Y'CbCr* colour encoding ranges array.
    is_legal : bool
        Whether the *Y'CbCr* colour encoding ranges array is legal.
    is_int : bool
        Whether the *Y'CbCr* colour encoding ranges array represents integer
        code values.

    Returns
    -------
    ndarray
        *Y'CbCr* colour encoding ranges array.

    Examples
    --------
    >>> YCbCr_ranges(8, True, True)
    array([ 16, 235,  16, 240])
    >>> YCbCr_ranges(8, True, False)  # doctest: +ELLIPSIS
    array([ 0.0627451...,  0.9215686...,  0.0627451...,  0.9411764...])
    >>> YCbCr_ranges(10, False, False)
    array([ 0. ,  1. , -0.5,  0.5])
    """

    if is_legal:
        ranges = np.array([16, 235, 16, 240])
        ranges *= 2 ** (bits - 8)
    else:
        ranges = np.array([0, 2 ** bits - 1, 0, 2 ** bits - 1])

    if not is_int:
        ranges = ranges.astype(np.float32) / (2 ** bits - 1)

    if is_int and not is_legal:
        ranges[3] = 2 ** bits

    if not is_int and not is_legal:
        ranges[2] = -0.5
        ranges[3] = 0.5

    return ranges.astype(np.float32)
def eotf_PQ_BT2100(E_p):
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* electro-optical
    transfer function (EOTF).

    The EOTF maps the non-linear *PQ* signal into display light.

    Parameters
    ----------
    E_p
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *PQ* space [0, 1].

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> eotf_PQ_BT2100(0.724769816665726)  # doctest: +ELLIPSIS
    779.9883608...
    """

    return eotf_ST2084(E_p, 10000)


def RGB_to_YCbCr(
    RGB,
    K= WEIGHTS_YCBCR["ITU-R BT.709"],
    in_bits= 10,
    in_legal= False,
    in_int= False,
    out_bits= 8,
    out_legal= True,
    out_int= False,
    **kwargs):


    Kr, Kb = K
    RGB_min, RGB_max = CV_range(in_bits, in_legal, in_int)
    Y_min, Y_max, C_min, C_max = YCbCr_ranges(out_bits, out_legal, out_int)

    RGB_float = RGB.astype(np.float32) - RGB_min
    RGB_float *= 1 / (RGB_max - RGB_min)
    R, G, B = RGB_float[:,:,0],RGB_float[:,:,1],RGB_float[:,:,2]

    Y = Kr * R + (1 - Kr - Kb) * G + Kb * B
    Cb = 0.5 * (B - Y) / (1 - Kb)
    Cr = 0.5 * (R - Y) / (1 - Kr)
    Y *= Y_max - Y_min
    Y += Y_min
    Cb *= C_max - C_min
    Cr *= C_max - C_min
    Cb += (C_max + C_min) / 2
    Cr += (C_max + C_min) / 2

    YCbCr = np.stack([Y, Cb, Cr],axis=2)

    return YCbCr
def YCbCr_to_RGB(YCbCr,
                 K=YCBCR_WEIGHTS['ITU-R BT.709'],
                 in_bits=8,
                 in_legal=True,
                 in_int=False,
                 out_bits=10,
                 out_legal=False,
                 out_int=False,
                 **kwargs):
    """
    Converts an array of *Y'CbCr* colour encoding values to the corresponding
    *R'G'B'* values array.

    Parameters
    ----------
    YCbCr : array_like
        Input *Y'CbCr* colour encoding array of integer or float values.
    K : array_like, optional
        Luma weighting coefficients of red and blue. See
        :attr:`colour.YCBCR_WEIGHTS` for presets. Default is
        *(0.2126, 0.0722)*, the weightings for *ITU-R BT.709*.
    in_bits : int, optional
        Bit depth for integer input, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Default is *8*.
    in_legal : bool, optional
        Whether to treat the input values as legal range. Default is *True*.
    in_int : bool, optional
        Whether to treat the input values as ``in_bits`` integer code values.
        Default is *False*.
    out_bits : int, optional
        Bit depth for integer output, or used in the calculation of the
        denominator for legal range float values, i.e. 8-bit means the float
        value for legal white is *235 / 255*. Ignored if ``out_legal`` and
        ``out_int`` are both *False*. Default is *10*.
    out_legal : bool, optional
        Whether to return legal range values. Default is *False*.
    out_int : bool, optional
        Whether to return values as ``out_bits`` integer code values. Default
        is *False*.

    Other Parameters
    ----------------
    in_range : array_like, optional
        Array overriding the computed range such as
        *in_range = (Y_min, Y_max, C_min, C_max)*. If ``in_range`` is
        undefined, *Y_min*, *Y_max*, *C_min* and *C_max* will be computed using
        :func:`colour.models.rgb.ycbcr.YCbCr_ranges` definition.
    out_range : array_like, optional
        Array overriding the computed range such as
        *out_range = (RGB_min, RGB_max)*. If ``out_range`` is undefined,
        *RGB_min* and *RGB_max* will be computed using :func:`colour.CV_range`
        definition.

    Returns
    -------
    ndarray
        *R'G'B'* array of integer or float values.

    Notes
    -----

    +----------------+-----------------------+---------------+
    | **Domain \\***  | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``YCbCr``      | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    +----------------+-----------------------+---------------+
    | **Range \\***   | **Scale - Reference** | **Scale - 1** |
    +================+=======================+===============+
    | ``RGB``        | [0, 1]                | [0, 1]        |
    +----------------+-----------------------+---------------+

    \\* This definition has input and output integer switches, thus the
    domain-range scale information is only given for the floating point mode.

    Warning
    -------
    For *Recommendation ITU-R BT.2020*, :func:`colour.YCbCr_to_RGB`
    definition is only applicable to the non-constant luminance implementation.
    :func:`colour.YcCbcCrc_to_RGB` definition should be used for the constant
    luminance case as per :cite:`InternationalTelecommunicationUnion2015h`.

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2011e`,
    :cite:`InternationalTelecommunicationUnion2015i`,
    :cite:`SocietyofMotionPictureandTelevisionEngineers1999b`,
    :cite:`Wikipedia2004d`

    Examples
    --------
    >>> YCbCr = np.array([502, 512, 512])
    >>> YCbCr_to_RGB(YCbCr, in_bits=10, in_legal=True, in_int=True)
    array([ 0.5,  0.5,  0.5])
    """

    YCbCr = YCbCr.astype(np.float32)
    Y, Cb, Cr = YCbCr[:,:,0], YCbCr[:,:,1], YCbCr[:,:,2],
    Kr, Kb = K
    Y_min, Y_max, C_min, C_max = YCbCr_ranges(in_bits, in_legal, in_int)
    RGB_min, RGB_max = CV_range(out_bits, out_legal, out_int)

    Y -= Y_min
    Cb -= (C_max + C_min) / 2
    Cr -= (C_max + C_min) / 2
    Y *= 1 / (Y_max - Y_min)
    Cb *= 1 / (C_max - C_min)
    Cr *= 1 / (C_max - C_min)
    R = Y + (2 - 2 * Kr) * Cr
    B = Y + (2 - 2 * Kb) * Cb
    G = (Y - Kr * R - Kb * B) / (1 - Kr - Kb)

    RGB = np.dstack([R, G, B])
    RGB *= RGB_max - RGB_min
    RGB += RGB_min
    RGB = np.round(RGB).astype(np.uint16) if out_int else RGB

    return RGB


