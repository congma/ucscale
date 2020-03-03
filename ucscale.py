"""Matrix-scaling implementation for the unit-consistent (i.e., diagonally
consistent) singular value decomposition and pseudoinverse.

The function dscale implements the dscale function, originally presented as
MatLab code, in the paper "A generalized matrix inverse that is consistent with
respect to diagonal transformations" by J. Uhlmann (doi: 10.1137/17M113890X).

The dscale function is vital for the evaluation of diagonally consistent
singular values.
"""
import numpy as np


_EPS = np.ldexp(2, -52)


def dscale(mat, tol=_EPS):
    """Evaluate diagonal scaling matrices and the positively scaled matrix
    for input real or complex matrix (2-dimensional array).

    Parameters
    ==========
        mat:
            Input real or complex matrix (two-dimensional array).

        tol:
            Tolerance parameter that controls the stopping criterion for the
            constrained matrix-optimization step.

    Return values
    =============
    The function returns a tuple 3 arrays (S, diag_l, diag_r), where
        S:
            Scaled matrix in the same shape as the input mat.
        diag_l, diag_r:
            One-dimensional arrays that contain the diagonal elements of the
            left- and right-scaling matrix respectively.

    Reference
    =========
        J. Uhlmann, A generalized matrix inverse that is consistent with
        respect to diagonal transformations. 2018, Soc. Ind. Appl. Math. J.
        Matrix Anal. Appl., 39(2):781--800 (doi: 10.1137/17M113890X)
    """
    # The implementation here makes extensive use of in-place ufuncs in order
    # to ease memory pressure and avoid allocating new array objects. The
    # in-place operations are safe if the iteration does not step over any
    # element more than once.
    sh_m, sh_n = np.shape(mat)  # Fails when the shape doesn't conform.
    out_ws = np.abs(mat, dtype=float)  # Eement-wise absolute value (real).
    mask_nz = out_ws != 0.0   # Mask for non-zero entries in the input.
    # "Sign" matrix with unit-magnitude elements of the input where nonzero,
    # and zero where zero.
    if np.iscomplexobj(mat):
        # Allocate array buffer and fill in the correct values at invalid
        # input locations at the same time.
        # Notice that numpy's sign() does not compute this matrix for complex
        # input.
        signs = np.zeros_like(mat)
        np.divide(mat, out_ws, where=mask_nz, out=signs)
    else:
        signs = np.sign(mat)
    # In-place replacement with element-wise logarithm, for the original abs
    # matrix is no longer needed. Base-2 logarithm (and later exponentiation)
    # takes advantage of easier maniputation.
    np.log2(out_ws, where=mask_nz, out=out_ws)
    # Column and row "valence" arrays, i.e. the (integer) number of non-zero
    # entries. Here, the boolean array mask_nz transparently gives integer
    # arrays when summed.
    c_valen = mask_nz.sum(axis=0)
    r_valen = mask_nz.sum(axis=1)
    # Buffers for the output scaling factors (real).
    diag_l = np.zeros(sh_m)    # Left.
    diag_r = np.zeros(sh_n)    # Right.
    # Column and row masks (boolean arrays) that restricts the manipulation to
    # non-zero divisors (see the "while" loop below).
    idxc = c_valen > 0
    idxr = r_valen > 0
    tol = np.abs(tol)
    xdel = np.inf
    # Solve the constrained matrix optimization problem that produces the
    # desired scaling.
    # Reference: U.G. Rothblum & S.A. Zenios, Scalings of matrices
    # satisfying line-product constraints and generalizations. 1992, Linear
    # Algebra Appl., 175, 159--175. (doi: 10.1016/0024-3795(92)90307-V)
    while xdel > tol:
        # Row reduction.
        # NOTE: Floating-point division by integer.
        pvec = out_ws[:, idxc].sum(axis=0) / c_valen[idxc]
        # NOTE: Implicit broadcast of pvec.
        out_ws[:, idxc] -= pvec * mask_nz[:, idxc]
        diag_r[idxc] -= pvec
        xdel = np.mean(np.abs(pvec))
        # Column reduction.
        pvec = out_ws[idxr, :].sum(axis=1) / r_valen[idxr]
        out_ws[idxr, :] -= pvec[:, None] * mask_nz[idxr, :]
        diag_l[idxr] -= pvec
        xdel += np.mean(np.abs(pvec))
    # Element-wise destructive replacement of the logarithms.
    np.exp2(diag_l, out=diag_l)
    np.exp2(diag_r, out=diag_r)
    np.exp2(out_ws, out=out_ws)
    # Re-apply the signs. Use the "sign" matrix signs as the in-place operand
    # because its type is already the most-promoted one (and correct for the
    # output). Notice that out_ws is real and can update a complex array
    # in-place. Therefore the array named signs at this stage does not merely
    # contain the signs.
    signs *= out_ws
    return signs, diag_l, diag_r
