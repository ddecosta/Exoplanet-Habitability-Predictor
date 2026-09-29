import numpy as np

def get_hz_boundaries(df):
    t_star = df['st_teff'] - 5780
    i_c = [1.7763, 2.1301e-4, 2.524e-9, -5.287e-13, -6.248e-17]
    o_c = [0.3207, 4.2856e-5, 5.287e-10, -5.967e-13, -9.100e-17]

    def calc_seff(c):
        return c[0] + c[1]*t_star + c[2]*t_star**2 + c[3]*t_star**3 + c[4]*t_star**4

    inner_au = np.sqrt((10**df['st_lum']) / calc_seff(i_c))
    outer_au = np.sqrt((10**df['st_lum']) / calc_seff(o_c))
    return inner_au, outer_au

def calculate_esi_component(val, reference, weight):
    return (1 - np.abs((val - reference) / (val + reference)))**weight

def calculate_esi_score(df):
    relative_flux = (10**df['st_lum']) / (df['pl_orbsmax']**2)
    esi_r = calculate_esi_component(df['pl_rade'], 1.0, 0.57)
    esi_f = calculate_esi_component(relative_flux, 1.0, 1.07)
    return np.sqrt(esi_r * esi_f)