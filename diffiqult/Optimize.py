''' This module was taken from scipy and modified to optimize
several parameters at a time with the BFGS Minimization '''
from __future__ import division, print_function, absolute_import


# Minimization routines

__all__ = ['fmin', 'fmin_powell', 'fmin_bfgs', 'fmin_ncg', 'fmin_cg',
           'fminbound', 'brent', 'golden', 'bracket', 'rosen', 'rosen_der',
           'rosen_hess', 'rosen_hess_prod', 'brute', 'approx_fprime',
           'line_search', 'check_grad', 'OptimizeResult', 'show_options',
           'OptimizeWarning']

__docformat__ = "restructuredtext en"

import warnings
import numpy
from scipy._lib.six import callable
from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
                   vectorize, asarray, sqrt, Inf, asfarray, isinf)
import numpy as np

from diffiqult.Linesearch import (line_search_wolfe1, line_search_wolfe2,
                         line_search_wolfe2 as line_search,
                         LineSearchWarning)
from scipy._lib._util import getargspec_no_self as _getargspec


# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}


class MemoizeJac(object):
    """ Decorator that caches the value gradient of function each time it
    is called. """
    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self.x = None

    def __call__(self, x, *args):
        self.x = numpy.asarray(x).copy()
        fg = self.fun(x, *args)
        self.jac = fg[1]
        return fg[0]

    def derivative(self, x, *args):
        if self.jac is not None and numpy.alltrue(x == self.x):
            return self.jac
        else:
            self(x, *args)
            return self.jac


class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"


class OptimizeWarning(UserWarning):
    pass


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)


def is_array_scalar(x):
    """Test whether `x` is either a scalar or an array scalar.

    """
    return np.size(x) == 1

_epsilon = sqrt(numpy.finfo(float).eps)


def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(numpy.abs(x))
    elif ord == -Inf:
        return numpy.amin(numpy.abs(x))
    else:
        return numpy.sum(numpy.abs(x)**ord, axis=0)**(1.0 / ord)


def rosen(x):
    """
    The Rosenbrock function.

    The function computed is::

        sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    f : float
        The value of the Rosenbrock function.

    See Also
    --------
    rosen_der, rosen_hess, rosen_hess_prod

    """
    x = asarray(x)
    r = numpy.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                  axis=0)
    return r


def rosen_der(x):
    """
    The derivative (i.e. gradient) of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the derivative is to be computed.

    Returns
    -------
    rosen_der : (N,) ndarray
        The gradient of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_hess, rosen_hess_prod

    """
    x = asarray(x)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = numpy.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der


def rosen_hess(x):
    """
    The Hessian matrix of the Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.

    Returns
    -------
    rosen_hess : ndarray
        The Hessian matrix of the Rosenbrock function at `x`.

    See Also
    --------
    rosen, rosen_der, rosen_hess_prod

    """
    x = atleast_1d(x)
    H = numpy.diag(-400 * x[:-1], 1) - numpy.diag(400 * x[:-1], -1)
    diagonal = numpy.zeros(len(x), dtype=x.dtype)
    diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
    H = H + numpy.diag(diagonal)
    return H


def rosen_hess_prod(x, p):
    """
    Product of the Hessian matrix of the Rosenbrock function with a vector.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Hessian matrix is to be computed.
    p : array_like
        1-D array, the vector to be multiplied by the Hessian matrix.

    Returns
    -------
    rosen_hess_prod : ndarray
        The Hessian matrix of the Rosenbrock function at `x` multiplied
        by the vector `p`.

    See Also
    --------
    rosen, rosen_der, rosen_hess

    """
    x = atleast_1d(x)
    Hp = numpy.zeros(len(x), dtype=x.dtype)
    Hp[0] = (1200 * x[0]**2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
    Hp[1:-1] = (-400 * x[:-2] * p[:-2] +
                (202 + 1200 * x[1:-1]**2 - 400 * x[2:]) * p[1:-1] -
                400 * x[1:-1] * p[2:])
    Hp[-1] = -400 * x[-2] * p[-2] + 200*p[-1]
    return Hp


def wrap_function(function, args, argnum=None, vec=None,print_out=False,name=None):
    '''This function returns the value of a function evaluated in args'''
    ### This is for counting the ncalls
    ncalls = [0]
    if function is None:
        return ncalls, None
    
    def function_wrapper(wrapper_args,print_out=False,name=None):
        ncalls[0] += 1
        if argnum is None:
            args1 = tuple([wrapper_args])+args
            return function(*(args1))
        else:
            split = []
            num = 1
            for i in vec:
               for j in i:
                   num *= j
               split.append(num)
            x = np.split(wrapper_args,split)
            args1 = []
            index = 0
            j = 0
            for i in range(len(argnum)+len(args)):
               if i in argnum:
                  args1.append(np.copy(np.reshape(x[argnum.index(i)],vec[j])))
                  j += 1
               else:
                  args1.append(args[index])
                  index += 1
            if print_out:
               args1[len(args1)-2] = True ## Activate flag for printint in autochem
               args1[len(args1)-3] = name ## Give the root for naming outputs.
            args1 = tuple(args1)
            value = function(*(args1))
            return value #np.transpose(value)
    return ncalls, function_wrapper

def wrap_functions(fprimes,args,argnum=None,vec=None):
    '''This function returns the value of a set of functions evaluated in args
    at this point it is used just for frimes'''
    ncalls = [0]
    grad_calls = []
    myfprime = []
    try:
      for fprime in fprimes:
          g_c, myfp = wrap_function(fprime, args,argnum,print_out=False, vec=vec)
          grad_calls.append(g_c)
          myfprime.append(myfp)
    except TypeError:
        return  wrap_function(fprimes, args, argnum, vec=vec,print_out=True)

    def gradients(x):
       ncalls[0] += 1
       grad = []
       for function in myfprime:
           grad.append(np.array(function(x)))
       if (len(grad)==1): 
          return np.reshape(grad,(len(grad[0]),))
       print(grad)
       grad= numpy.concatenate(grad)
       return np.reshape(grad,(grad.shape[0],))
    return ncalls,gradients

def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None):
    """
    Minimize a function using the downhill simplex algorithm.

    This algorithm only uses function values, not derivatives or second
    derivatives.

    Parameters
    ----------
    func : callable func(x,*args)
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func, i.e. ``f(x,*args)``.
    callback : callable, optional
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    xtol : float, optional
        Relative error in xopt acceptable for convergence.
    ftol : number, optional
        Relative error in func(xopt) acceptable for convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : number, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        Set to True if fopt and warnflag outputs are desired.
    disp : bool, optional
        Set to True to print convergence messages.
    retall : bool, optional
        Set to True to return list of solutions at each iteration.

    Returns
    -------
    xopt : ndarray
        Parameter that minimizes function.
    fopt : float
        Value of function at minimum: ``fopt = func(xopt)``.
    iter : int
        Number of iterations performed.
    funcalls : int
        Number of function calls made.
    warnflag : int
        1 : Maximum number of function evaluations made.
        2 : Maximum number of iterations reached.
    allvecs : list
        Solution at each iteration.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Nelder-Mead' `method` in particular.

    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.

    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does.

    References
    ----------
    .. [1] Nelder, J.A. and Mead, R. (1965), "A simplex method for function
           minimization", The Computer Journal, 7, pp. 308-313

    .. [2] Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
           Respectable", in Numerical Analysis 1995, Proceedings of the
           1995 Dundee Biennial Conference in Numerical Analysis, D.F.
           Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
           Harlow, UK, pp. 191-208.

    """
    opts = {'xtol': xtol,
            'ftol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'return_all': retall}

    res = _minimize_neldermead(func, x0, args, callback=callback, **opts)
    if full_output:
        retlist = res['x'], res['fun'], res['nit'], res['nfev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']



def _approx_fprime_helper(xk, f, epsilon, args=(), f0=None):
    """
    See ``approx_fprime``.  An optional initial function value arg is added.

    """
    if f0 is None:
        f0 = f(*((xk,) + args))
    grad = numpy.zeros((len(xk),), float)
    ei = numpy.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad

def algopy_wrapper(function):
    """Returnfunction to obtain the gradient of the function
    Parameters
    ----------
    f : callable
        The function of which to determine the gradient (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a scalar, the value of the
        function at `xk`.
    Returns
    -------
    grad_function : function
        The partial derivatives of `f` to `xk`.

    """
    from algopy import UTPM
    def algopy_fprime(xk, *args):
        """ Evaluates the gradient of the function 
        Parameters
        ----------
        xk : array_like
             The coordinate vector at which to determine the gradient of `f`.
        Returns
        -------
        grad : ndarray
              The partial derivatives of `f` to `xk`.
        """
        var = UTPM.init_jacobian(xk)
        grad = UTPM.extract_jacobian(function(*(tuple([var])+args)))
        return grad
    return algopy_fprime

def approx_fprime(xk, f, epsilon, *args):
    """Finite-difference approximation of the gradient of a scalar function.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        The function of which to determine the gradient (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a scalar, the value of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives.  If an array, should contain one value per element of
        `xk`.
    \*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    grad : ndarray
        The partial derivatives of `f` to `xk`.

    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    The main use of `approx_fprime` is in scalar function optimizers like
    `fmin_bfgs`, to determine numerically the Jacobian of a function.

    Examples
    --------
    >>> from scipy import optimize
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(float).eps)
    >>> optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004198])

    """
    return _approx_fprime_helper(xk, f, epsilon, args=args)


def check_grad(func, grad, x0, *args, **kwargs):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func : callable ``func(x0, *args)``
        Function whose derivative is to be checked.
    grad : callable ``grad(x0, *args)``
        Gradient of `func`.
    x0 : ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args : \*args, optional
        Extra arguments passed to `func` and `grad`.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        ``sqrt(numpy.finfo(float).eps)``, which is approximately 1.49e-08.

    Returns
    -------
    err : float
        The square root of the sum of squares (i.e. the 2-norm) of the
        difference between ``grad(x0, *args)`` and the finite difference
        approximation of `grad` using func at the points `x0`.

    See Also
    --------
    approx_fprime

    Examples
    --------
    >>> def func(x):
    ...     return x[0]**2 - 0.5 * x[1]**3
    >>> def grad(x):
    ...     return [2 * x[0], -1.5 * x[1]**2]
    >>> from scipy.optimize import check_grad
    >>> check_grad(func, grad, [1.5, -1.5])
    2.9802322387695312e-08

    """
    step = kwargs.pop('epsilon', _epsilon)
    if kwargs:
        raise ValueError("Unknown keyword arguments: %r" %
                         (list(kwargs.keys()),))
    return sqrt(sum((grad(x0, *args) -
                     approx_fprime(x0, func, step, *args))**2))


def approx_fhess_p(x0, p, fprime, epsilon, *args):
    f2 = fprime(*((x0 + epsilon*p,) + args))
    f1 = fprime(*((x0,) + args))
    return (f2 - f1) / epsilon


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, xtol=1e-4,
                         **kwargs):
    '''
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    '''
    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval, xtol=xtol,
                             **kwargs)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk, 
                                     old_fval, old_old_fval,xtol=xtol)

    if ret[0] is None:
        raise _LineSearchError()

    return ret


def fmin_bfgs(f, x0, fprime=None, args=(), argnum=None, gtol=1e-5, norm=Inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None):
    '''
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable f'(x,*args), optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    gtol : float, optional
        Gradient norm must be less than gtol before successful termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If fprime is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True,return fopt, func_calls, grad_calls, and warnflag
        in addition to xopt.
    disp : bool, optional
        Print convergence message if True.
    retall : bool, optional
        Return a list of results at each iteration if True.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e. the inverse hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
    allvecs  :  list
        `OptimizeResult` at each iteration.  Only returned if retall is True.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'BFGS' `method` in particular.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS)

    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.
    This function was originally taken and modified from scipy
    '''
    opts = { 'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_bfgs(f, x0, args, argnum, fprime, callback=callback,gtol=gtol, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['jac'], res['hess_inv'],
                   res['nfev'], res['njev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_bfgs(fun, x0, args=(), argnum=None, jac=None, callback=None, name=None,
                   gtol=1e-1, etol=1e-5, norm=Inf, eps=_epsilon, maxiter=30,
                   disp=False, return_all=False,xtol_linew=1e-14,verbose=1,print_out=False,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.
    etol : float
        Difference between steps for succesful termination.
    verbose : bool
        Set true print steps values and gradients on screen.
    print_out: bool
        Set True to print molden files of each optimization step. 
    """
    _check_unknown_options(unknown_options)
    f = fun
    fprimes = jac
    epsilon = eps
    retall = return_all

    vec = []
    for i in x0:
       vec.append(i.shape)

    x0 = np.concatenate([asarray(i).flatten() for i in x0]).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200
               
    func_calls, f = wrap_function(f, args,argnum, vec=vec)

    if fprimes is None:
       ### numerical derivative
       grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
       print(myfprime)
    else:
       grad_calls, myfprime = wrap_functions(fprimes, args,argnum, vec=vec)

    
    val = 100.0
    k = 0 
    N = len(x0)
    I = numpy.eye(N, dtype=int)
    Hk = I
    xk = x0
    if retall:
        allvecs = [x0]
    sk = [2 * gtol]
    warnflag = 0

    old_fval = f(xk,print_out=print_out,name=str(name)+'-BFGS_step_'+str(k))
    gfk = myfprime(x0)
    old_old_fval = None

    # Checking convergence with gradient criteria
    gnorm = vecnorm(gfk, ord=np.inf)

    while (gnorm > gtol) and (k < maxiter):
        if verbose:
           print ('STEP: ',k, old_fval)
           print ('x: ',xk)
           print ('gfk: ',gfk)
        pk = -numpy.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, xtol=xtol_linew)
        except _LineSearchError:
            warnflag = 2
            break


        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break
        if (abs(val-old_fval) <= etol):
            break

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rhok
        A2 = I - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * sk[:, numpy.newaxis] *
                                                 sk[numpy.newaxis, :])
        if print_out:
           # An extra evaluation for printing molden file
           old_fval = f(xk,print_out=print_out,name=str(name)+'-BFGS_step_'+str(k))
  
    fval = old_fval
    if verbose:
        print ('Last step: ',k, old_fval)
        print ('x: ',xk)
        print ('gfk: ',gfk)

    if np.isnan(fval):
        # This can happen if the first call to f returned NaN;
        # the loop is then never entered.
        warnflag = 2

    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    elif k >= maxiter:
        warnflag = 1
        old_fval = f(xk,print_out=True,name=str(name)+'-BFGS_step_'+str(k))

        msg = _status_message['maxiter']
        if disp:
            print("Warning: " + msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % func_calls[0])
            print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result



def show_options(solver=None, method=None, disp=True):
    """
    Show documentation for additional options of optimization solvers.

    These are method-specific options that can be supplied through the
    ``options`` dict.

    Parameters
    ----------
    solver : str
        Type of optimization solver. One of 'minimize', 'minimize_scalar',
        'root', or 'linprog'.
    method : str, optional
        If not given, shows all methods of the specified solver. Otherwise,
        show only the options for the specified method. Valid values
        corresponds to methods' names of respective solver (e.g. 'BFGS' for
        'minimize').
    disp : bool, optional
        Whether to print the result rather than returning it.

    Returns
    -------
    text
        Either None (for disp=False) or the text string (disp=True)

    Notes
    -----
    The solver-specific methods are:

    `scipy.optimize.minimize`

    - :ref:`BFGS        <optimize.minimize-bfgs>`

    """
    import textwrap

    doc_routines = {
        'minimize': (
            ('bfgs', 'scipy.optimize.optimize._minimize_bfgs'),
        ),
    }

    if solver is None:
        text = ["\n\n\n========\n", "minimize\n", "========\n"]
        text.append(show_options('minimize', disp=False))
        text.extend(["\n\n===============\n", "minimize_scalar\n",
                     "===============\n"])
    else:
        solver = solver.lower()
        if solver not in doc_routines:
            raise ValueError('Unknown solver %r' % (solver,))

        if method is None:
            text = []
            for name, _ in doc_routines[solver]:
                text.extend(["\n\n" + name, "\n" + "="*len(name) + "\n\n"])
                text.append(show_options(solver, name, disp=False))
            text = "".join(text)
        else:
            methods = dict(doc_routines[solver])
            if method not in methods:
                raise ValueError("Unknown method %r" % (method,))
            name = methods[method]

            # Import function object
            parts = name.split('.')
            mod_name = ".".join(parts[:-1])
            __import__(mod_name)
            obj = getattr(sys.modules[mod_name], parts[-1])

            # Get doc
            doc = obj.__doc__
            if doc is not None:
                text = textwrap.dedent(doc).strip()
            else:
                text = ""

    if disp:
        print(text)
        return
    else:
        return text


def main():
    import time

    times = []
    algor = []
    x0 = np.array([0.8, 1.2, 0.7])

    print()
    print("BFGS Quasi-Newton")
    print("=================")
    start = time.time()
    x = fmin_bfgs(rosen, x0, fprime=rosen_der,maxiter=80)
    print(x)
    times.append(time.time() - start)
    algor.append('BFGS Quasi-Newton\t')

    print()
    print("BFGS approximate gradient")
    print("=========================")
    start = time.time()
    x = fmin_bfgs(rosen, x0, gtol=1e-4, maxiter=100)
    print(x)
    times.append(time.time() - start)
    algor.append('BFGS without gradient\t')



    print()
    print("BFGS algopy gradient")
    print("=========================")
    start = time.time()
    x = fmin_bfgs(rosen, x0, fprime=algopy_wrapper(rosen),gtol=1e-4, maxiter=100)
    print(x)
    times.append(time.time() - start)
    algor.append('BFGS without gradient\t')

if __name__ == "__main__":
    main()
