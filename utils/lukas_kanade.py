import numpy as np
from scipy import interpolate
from utils import se3


def calcResiduals(IRef, DRef, I, D, xi, K, norm_param, use_hubernorm):
    T = se3.se3Exp(xi)
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    KInv = np.linalg.inv(K)

    xImg = np.zeros_like(IRef) - 10
    yImg = np.zeros_like(IRef) - 10
    Depth1 = np.zeros_like(DRef)

    for x in range(IRef.shape[0]):
        for y in range(IRef.shape[1]):
            p = DRef[x, y] * np.dot(KInv, np.array([x, y, 1]))

            pTrans = np.dot(K, np.dot(R, p) + t)
            Depth1[x, y] = pTrans[2]

            if (pTrans[2] > 0) & (DRef[x, y] > 0):
                xImg[x, y] = pTrans[0] / pTrans[2]
                yImg[x, y] = pTrans[1] / pTrans[2]

    # interpolation method 1
    # xx, yy = np.mgrid[:I.shape[0], :I.shape[1]]
    # interp_I = interpolate.griddata(np.stack([xx.ravel(), yy.ravel()]).T, I.ravel(), (xImg.real, yImg.real))

    # interpolation method 2
    xx = np.arange(0, I.shape[0])
    yy = np.arange(0, I.shape[1])
    f = interpolate.RegularGridInterpolator((xx, yy), I, bounds_error=False)
    # g = interpolate.RegularGridInterpolator((xx, yy), D, bounds_error=False)
    interp_I = f((xImg.real.ravel(), yImg.real.ravel())).reshape(I.shape)
    # interp_D = g((xImg.real.ravel(), yImg.real.ravel())).reshape(D.shape)

    # residuals = IRef - interp_I + Depth1 - interp_D
    residuals = IRef - interp_I
    weights = 0 * residuals + 1

    if use_hubernorm:
        idx = np.where(abs(residuals) > norm_param)
        weights[idx] = norm_param / abs(residuals[idx])
    else:
        weights = 2 / (1 + residuals ** 2 / norm_param ** 2) ** 2

    residuals = residuals.reshape(I.shape[0] * I.shape[1], -1, order='F')
    weights = weights.reshape(I.shape[0] * I.shape[1], -1, order='F')

    return residuals, weights


def deriveJacobianNumeric(IRef, DRef, I, D, xi, K, norm_param, use_hubernorm):
    eps = 1e-6
    Jac = np.zeros([I.shape[0] * I.shape[1], 6])
    residuals, weights = calcResiduals(IRef, DRef, I, D, xi, K, norm_param, use_hubernorm)

    for j in range(6):
        epsVec = np.zeros([6, 1])
        epsVec[j] = eps

        # multiply epsilon from left onto the current estimate
        xiPerm = se3.se3Log(se3.se3Exp(epsVec) @ se3.se3Exp(xi))
        residual_xiPerm, _ = calcResiduals(IRef, DRef, I, D, xiPerm, K, norm_param, use_hubernorm)
        Jac[:, j] = (residual_xiPerm - residuals).ravel() / eps

    return Jac, residuals, weights


def do_alignment(IRef, DRef, K, I, D, xi, norm_param, use_hubernorm):
    errLast = 1e10
    hessian_m = np.identity(6)
    hessian_m_new = hessian_m.copy()
    xi_new = xi.copy()
    for j in range(20):
        Jac, residual, weights = deriveJacobianNumeric(IRef, DRef, I, D, xi_new, K, norm_param, use_hubernorm)

        notValid = np.isnan(np.sum(Jac, 1) + residual.ravel())
        residual[notValid] = 0
        Jac[notValid] = 0
        weights[notValid] = 0

        hessian_m = Jac.T @ (np.tile(weights, (1, 6)) * Jac)
        upd = - np.linalg.inv(hessian_m) @ Jac.T @ (weights * residual)
        xi = se3.se3Log(se3.se3Exp(upd) @ se3.se3Exp(xi_new))
        err = np.mean(residual ** 2)
        # print('step:', j, 'err:', err, 't', xi)

        if err > errLast:
            break

        errLast = err
        xi_new = xi.copy()
        hessian_m_new = hessian_m.copy()
    return xi_new, hessian_m_new
