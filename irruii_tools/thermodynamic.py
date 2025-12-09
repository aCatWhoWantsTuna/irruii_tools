import numpy as np
from . import numerical as num

Lv = 2.5e6     # J/kg
Rv = 461       # J/kg/K
Rd = 287.05       # J/kg/K
Cp = 1004.6     # J/kg/K
eps = Rd / Rv
T0  = 273.15   # K
es0 = 6.112    # hPa
g = 9.806

def theta(T, p):
    """
    Dry potential temperature th [K]
    INPUT: T[K], p[hPa]
    """
    return T * (1000 / p) ** (Rd / Cp)

def qvs(T, p):
    """
    Saturation specific humidity qvs [kg/kg]
    INPUT: T[K], p[hPa]
    """
    es = es0 * np.exp(Lv / Rv * (1/T0 - 1/T))  # hPa
    return eps * es / p

def Tc_Bolton(T, Td):
    """Bolton 1980 corrected temperature Tc[K]"""
    return 1.0 / (1.0/(Td-56.0) + np.log(T/Td)/800.0) + 56.0

def Td(p, qv):
    """Estimated Td [K]"""
    r = qv / (1 - qv)
    e = r * p / (eps + r)
    return 243.5 * np.log(e/6.112) / (17.67 - np.log(e/6.112)) + 273.15

def theta_e(T, p, qv, Td_=None):
    """
    Equivalent potential temperature theta_e
    if no Td, use surface T[K] to calculate it
    """
    th = theta(T, p)

    if Td_ is None:  # find Td
        Td_ = Td(p, qv)

    Tc = Tc_Bolton(T, Td_)
    return th * np.exp(Lv * qv / Cp / Tc)

def theta_es(T, p, Td=None):
    """
    INPUT: T[K], p[hPa]
    """
    qvs_val = qvs(T, p)
    return theta_e(T, p, qvs_val, T)

def Tv(T, qv):
    """Virtual temperature"""
    return T * (1 + 0.608 * qv)

def p_to_z(p, T, qv=0, z0=0):
    """
    input: arr, output: arr
    """
    p = np.asarray(p)
    z = np.zeros_like(p)
    if p[0] < p[-1]:
        p = p[::-1]
        T = T[::-1]
        if np.any(qv):
            qv = qv[::-1]
    z[0] = z0
    Tv = T * (1 + 0.608 * qv)

    for i in range(1, len(p)):
        log_p_local = 0.5*(np.log(p[i-1]) + np.log(p[i]))
        p_local = np.exp(log_p_local)
        Tv_layer = interp_profile(p,Tv, p_local)
        z[i] = z[i-1] + Rd * Tv_layer / g * np.log(p[i-1]/p[i])

    return z
    
def interp_profile(x, y, x_target, log_x=True):
    """
    Interpolate profile y(x) to a new point x_target.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if log_x:
        x_val = np.log(np.asarray(x))
        x_new = np.log(x_target)
    else:
        x_val = np.asarray(x)
        x_new = x_target

    if x_val[0] > x_val[-1]:
        x_val = -x_val
        x_new = -x_new
    
    return np.interp(x_new, x_val, y)
    
def dry_adiabatic(T0, p_prof):
    T_parcel = theta(T0, p_prof[0]) * (p_prof / 1000) ** (Rd/Cp)
    return T_parcel

def moist_adiabatic(T_prev, p_prev, p_next, qv_prev, tol=1e-6, max_iter=100):

    T_temp = dry_adiabatic(T_prev, np.array([p_prev, p_next]))[-1]
    T_min = T_temp
    T_max = 350.
    
    for i in range(max_iter):
        T_mid = 0.5 * (T_min + T_max)
        p_mid = np.exp(0.5 * (np.log(p_prev)+np.log(p_next)))
        qvs_mid = qvs(T_mid, p_mid)
        err = (qv_prev - qvs_mid) * Lv/ Cp - (T_mid - T_temp)
        if abs(err)<tol:
            break
        elif err>0:
            T_min = T_mid
        else:
            T_max = T_mid
    return T_mid, qvs_mid


def LCL_idx_p(T, p, qv,):
    """
    Find LCL index with optional interpolation for more precise height.

    INPUT:
        T[K]  : environmental temperature
        p[hPa] : pressure array
        qv[kg/kg] : water vapor mixing ratio
        z0[m] : surface height

    OUTPUT:
        idx_lcl    : the first layer idx that qv >= qvs(not accurate)
        p_lcl[hPa] : pressure at LCL
        z_lcl[m]   : height at LCL
    """
    if p[0]<p[1]:
        p= p[::-1]
        T = T[::-1]
        qv = qv[::-1]
    # Dry adiabat parcel temperature
    T_parcel = dry_adiabatic(T[0], p)
    qvs_prof = qvs(T_parcel, p)
    qv_prof = np.full_like(qvs_prof, qv[0])

    # Find first level where parcel becomes saturated
    idx = np.where(qv[0] >= qvs_prof)[0]
    if len(idx) == 0:
        return None, None

    idx_lcl = idx[0] # approximately
    p_lcl = p[idx_lcl]
    # 
    if idx_lcl > 0:
        qv_lower = qv_prof[idx_lcl]
        qv_upper = qvs_prof[idx_lcl]

        # If LCL is not at the first level, do linear interpolation for precision
        if qv_upper != qv_lower: 
            # Linear interpolation for exact LCL pressure
            p_lcl, _ = num.find_cross_point(p, qv_prof, qvs_prof,)
    # print(idx_lcl, p_lcl)

    return idx_lcl, p_lcl



def parcel_profile(T_env, p_env, qv_env, z0 = 0):
    """
    Generate parcel ascent profile:
        - dry adiabat below LCL
        - moist adiabat above LCL

    INPUT:
        T_env  : environmental temperature [K]
        p_env  : pressure [hPa]
        qv_env : environmental qv [kg/kg]
        z      : optional height array

    OUTPUT:
        T_parcel    : parcel temperature profile [K]
        qv_parcel   : parcel humidity profile [kg/kg]
        LCL_index
    """
    if p_env[0]<p_env[1]:
        p_env = p_env[::-1]
        T_env = T_env[::-1]
        qv_env = qv_env[::-1]
    # 1. Below LCL: dry adiabat
    T_parcel = dry_adiabatic(T_env[0], p_env)
    qvs_prof = qvs(T_parcel, p_env)
    qv_parcel = np.full_like(qvs_prof, qv_env[0])
 
    # 2. Above LCL: moist adiabat
    idx_lcl, p_lcl = LCL_idx_p(T_env, p_env, qv_env)

    if idx_lcl is not None:
        T_lcl = dry_adiabatic(T_parcel[idx_lcl-1], np.array([p_env[idx_lcl-1], p_lcl]))[1]
        qv_lcl = qv_env[0]
        for i in range(idx_lcl, len(T_parcel)):
            if i == idx_lcl:
                T_parcel[i], qv_parcel[i] = moist_adiabatic(T_lcl, p_lcl, p_env[i],qv_lcl)
            else:
                T_parcel[i], qv_parcel[i] = moist_adiabatic(T_parcel[i-1], p_env[i-1], p_env[i],qv_parcel[i-1])
    return T_parcel, qv_parcel, idx_lcl



def find_EL_LFC_CIN_CAPE(T_env, p_env, qv_env, z0=0, Need_Precise_val = False):
    """
    Find LFC, EL indices using cross-point (B=0) and calculate CIN/CAPE.
    
    INPUT:
        T_env, p_env, qv_env, z0    
    OUTPUT:
        LFC_idx, EL_idx, CIN, CAPE, (LFC_pressure_precise, EL_pressure_precise if Need_Precise_val)
    """
    if p_env[0]<p_env[1]:
        p_env = p_env[::-1]
        T_env = T_env[::-1]
        qv_env = qv_env[::-1]

    T_parcel, qv_parcel, _ = parcel_profile(T_env, p_env, qv_env)
    z_env = p_to_z(p_env, T_env, qv_env, z0)

    # 2. virtual potential temperature & buoyancy
    Tv_env = Tv(T_env, qv_env)
    Tv_par = Tv(T_parcel, qv_parcel)
    B = g * (Tv_par - Tv_env) / Tv_env
    
    for i in range(len(B)):
        print(B[i], p_env[i])


    # --------------------------
    # 1. LFC: first B>0, use cross-point
    LCL_idx,_ = LCL_idx_p(T_env, p_env, qv_env)
    if LCL_idx == None:
        if Need_Precise_val:
            return None, None, 0, 0, None, None
        else: return None, None, 0,0

    LCL_idx_lower = max(LCL_idx - 1, 0)

    # ---- 從修正後的位置開始找 LFC ----
    LFC_precise, LFC_idx = num.find_cross_point(
        p_env[LCL_idx_lower:], 
        np.zeros_like(B[LCL_idx_lower:]), 
        B[LCL_idx_lower:], 
        log_x=True)

    if LFC_idx is None:
        if Need_Precise_val:
            return None, None, 0, 0, None, None
        else: return None, None, 0,0
        # return None, None, 0.0, 0.0

    LFC_idx += LCL_idx_lower

    # --------------------------
    # 2. EL: first B<0 above LFC
    EL_precise, EL_idx = num.find_cross_point(p_env[LFC_idx:], 
                                              np.zeros_like(B[LFC_idx:]), 
                                              B[LFC_idx:], 
                                              log_x = True)
    if EL_idx is None:
        EL_idx = len(B) - 1
    else:
        EL_idx += LFC_idx  # adjust relative index
    # print(B, B.shape, EL_idx)


    # # --------------------------
    # # 3. CIN: integrate B<0 from surface to LFC
    # CIN = 0.0
    # if LFC_idx <=1:
    #     CIN = 0.0
    # for k in range(1, int(LFC_idx)):
    #     dz = z_env[k] - z_env[k-1]
    #     if B[k] < 0:
    #         CIN += (B[k]) * dz

    # # --------------------------
    # # 4. CAPE: integrate B>0 from LFC to EL
    # CAPE = 0.0
    # for k in range(int(LFC_idx), int(EL_idx)):
    #     dz = z_env[k] - z_env[k-1]
    #     if B[k] > 0:
    #         CAPE += B[k] * dz
    z_CIN = np.concatenate([z_env[:LFC_idx], [np.interp(LFC_precise, p_env[::-1], z_env[::-1])]])
    
    B_CIN = np.interp(z_CIN, z_env, B)
    mask = B_CIN < 0
    CIN = np.trapz(B_CIN[mask], z_CIN[mask])
    z_CAPE = np.concatenate([[z_CIN[-1]], z_env[LFC_idx:EL_idx], [np.interp(EL_precise, p_env[::-1], z_env[::-1])]])
    B_CAPE = np.interp(z_CAPE, z_env, B)

    mask = B_CAPE > 0
    CAPE = np.trapz(B_CAPE[mask], z_CAPE[mask])

    if Need_Precise_val:
        return LFC_idx, EL_idx, CIN, CAPE, LFC_precise, EL_precise
    else:
        return LFC_idx, EL_idx, CIN, CAPE

def project_wind_speed_north_ref(u_wind,  v_wind,  angle_deg: float) -> float:
    """
    angle 0 is to the north.

    Input:
        u_wind (float): + to the east
        v_wind (float): + to the north
        angle_deg (float): the angle i wanna project

    """
    u_wind, v_wind = np.array([u_wind, v_wind], dtype=float)
    angle_rad = np.deg2rad(angle_deg)
    u_proj = np.sin(angle_rad) 
    v_proj = np.cos(angle_rad)
    

    projected_speed = u_wind * u_proj + v_wind * v_proj
    
    return projected_speed