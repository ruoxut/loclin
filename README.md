# loclin
## The local linear ridged estimator for E(Y|X) and the local quadratic ridged estimator for E'(Y|X) (the first derivative) with one-d covariate.


## *Features*:
1. Fully data-driven bandwidth selection using the pulg-in idea with cross validation pilot bandwidth.
2. There is a ridge term in the denomiator of the final estimate to prevent a too small denominator.
3. The kernel is Gaussian.

## Input:
X: n*1 covariate vector

Y: n*1 outcome vector

[a,b]: the interval for which we estimate E(Y|X), [q_0.025 and q_0.975] as default

## Output:
t: the vector where we evaluate the estimator

f1: the vector of estimated E(Y|X=x)

f2: the vector of estimated E'(Y|X=x)

## References:
1. Fan, J and Gijbels, I. (1996). Local polynomial modelling and its applications, vol 66. CRC Press.
2. Lin, Z and Yao, F. (2020). Functional regression on the manifold with contamination. Biometrika.
