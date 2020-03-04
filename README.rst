ucscale
=======

Matrix-scaling implementation for the unit-consistent (i.e., diagonally
consistent) singular value decomposition and pseudoinverse.

The function ``dscale`` implements the identically named function, originally
presented as MatLab code, in the paper *A Generalized Matrix Inverse That Is
Consistent with Respect to Diagonal Transformations* by J. Uhlmann [UHL2018]_.
Further information can be found in the paper *Scalings of Matrices Satisfying
Line-Product Constraints and Generalizations* by U.G. Rothblum and S.A. Zenios
[ROT1992]_.

The dscale function is vital for the computation of diagonally consistent
singular values.


.. [UHL2018] J. Uhlmann. A generalized matrix inverse that is consistent with
   respect to diagonal transformations. 2018, Soc. Ind. Appl. Math. J. Matrix
   Anal. Appl., 39(2):781--800 (doi: 10.1137/17M113890X)
.. [ROT1992] U.G. Rothblum & S.A. Zenios. Scalings of matrices satisfying
   line-product constraints and generalizations. 1992, Linear Algebra Appl.,
   175, 159--175. (doi: 10.1016/0024-3795(92)90307-V)
