# numerical.py
import numpy as np
from scipy.integrate import solve_ivp
# from scipy.interpolate import interp1d


def rk4_step(f, y, x, dx):
    k1 = f(y, x) * dx
    k2 = f(y + 0.5*k1, x + 0.5*dx) * dx
    k3 = f(y + 0.5*k2, x + 0.5*dx) * dx
    k4 = f(y + k3, x + dx) * dx
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

def integrate_rk4(f, y0, x):

    y = np.zeros_like(x, dtype=float)
    y[0] = y0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        y[i] = rk4_step(f, y[i-1], x[i-1], dx)
    return y

def integrate_rk_adaptive(f, y0, x_span, t_eval=None, rtol=1e-6, atol=1e-9):
    """
    RK45, SciPy solve_ivp
    f: dy/dx = f(y, x)
    y0: initial
    x_span: (x_start, x_end)
    t_eval: x array (output)
    """
    sol = solve_ivp(f, x_span, [y0], method='RK45', t_eval=t_eval, rtol=rtol, atol=atol)
    return sol.y[0], sol.t

def finite_difference(f, x, dx=1e-3, method='central'):
    """
    first derivative
    f: function or array
    x: grid array
    method: 'forward', 'backward', 'central'
    """
    f = np.asarray(f)
    df = np.zeros_like(f)
    if method == 'forward':
        df[:-1] = (f[1:] - f[:-1]) / (x[1:] - x[:-1])
        df[-1] = df[-2]
    elif method == 'backward':
        df[1:] = (f[1:] - f[:-1]) / (x[1:] - x[:-1])
        df[0] = df[1]
    else:  # central
        df[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
        df[0] = df[1]
        df[-1] = df[-2]
    return df



def insert_point_1d(x, y, x_new, log_x=False):
    """
    Insert a new x value and interpolate y (1D only).
    
    INPUT
        x       : original x array
        y       : original y array (1D)
        x_new   : new x to insert
        log_x   : whether to interpolate in log(x) space (e.g., pressure)
    
    OUTPUT
        x_out   : new x array with inserted value
        y_out   : new y array with interpolated value
        idx     : index where x_new was inserted
    """
    x = np.array(x, dtype = float)
    y = np.array(y, dtype = float)

    if log_x == True:
        x_val = np.log(x)
        x_new_val = np.log(x_new)
    else:
        x_val = x
        x_new_val = x_new

    # find insert position
    if x[-1] - x[0] < 0: 
        x_val, x_new_val = -x_val, -x_new_val
    idx = np.searchsorted(x_val, x_new_val)
    # interpolate
    y_new = np.interp(x_new_val, x_val, y)
    x_out = np.insert(x, idx, x_new)
    y_out = np.insert(y, idx, y_new)

    return x_out, y_out, idx

def find_cross_point(x, y1, y2, log_x=False):
    """
    find crossection (x_new)
    
    INPUT:
        x      : for checking position
        y1, y2 : conpaired vars (1D array)
        log_x  
        
    OUTPUT:
        x_new  : x value in crossection of y1 & y2 
        idx    : idx before crossing 
    """
    x = np.asarray(x, dtype = float)
    y1 = np.asarray(y1, dtype = float)
    y2 = np.asarray(y2, dtype = float)
    
    dy = y1 - y2
    sign_change = np.where(dy[:-1] * dy[1:] <= 0)[0]

    if len(sign_change) == 0:
        return None, None  
    
    idx_prev = sign_change[0]
    # print(i)
    
    x0, x1 = x[idx_prev], x[idx_prev+1]
    y0, y1_ = dy[idx_prev], dy[idx_prev+1]
    
    if log_x == True:
        log_x0, log_x1 = np.log(x0), np.log(x1)
        log_x_new = log_x0 - y0 * (log_x1 - log_x0) / (y1_- y0)
        
        x_new = np.exp(log_x_new)
    else:
        x_new = x0 - y0 * (x1 - x0) / (y1_ - y0)
    
    idx_next = idx_prev +1
    return x_new, idx_next

def remove_point(y, idx):
    y_new = np.delete(y, idx)
    return y_new

