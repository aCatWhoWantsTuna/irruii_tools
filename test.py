import numpy as np

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


if __name__ == "__main__":
    u_example = [7.07,6,5,4,]
    v_example = [7.07,7.2,7.5,7.8]

    angle1 = 45.
    proj1 = project_wind_speed_north_ref(u_example, v_example, angle1)
    print(proj1)