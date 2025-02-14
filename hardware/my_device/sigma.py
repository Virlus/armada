import numpy as np
import time
from scipy.spatial.transform import Rotation as R

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sigma_sdk"))
import sigma7

class Sigma7:
    def __init__(self, pos_scale=5, width_scale=1000) -> None:
        self.pos_scale = pos_scale
        self.width_scale = width_scale
        self.start_sigma()
        sig, px, py, pz, oa, ob, og, pg, matrix = sigma7.drdGetPositionAndOrientation()
        self.init_p = np.array([py,-px,pz])
        self.init_r = np.array([oa, ob, og])

    def start_sigma(self):
        sigma7.drdOpen()
        sigma7.drdAutoInit()
        print('starting sigma')
        sigma7.drdStart()
        sigma7.drdRegulatePos(on = False)
        sigma7.drdRegulateRot(on = False)
        sigma7.drdRegulateGrip(on = False)
        print('sigma ready')
    
    def get_control(self):
        sig, px, py, pz, oa, ob, og, pg, matrix = sigma7.drdGetPositionAndOrientation()
        curr_p = np.array([py,-px,pz])
        curr_r = np.array([oa, ob, og])

        diff_p = curr_p - self.init_p
        diff_r = curr_r - self.init_r
        diff_p = diff_p * self.pos_scale
        width = pg / -0.027 * self.width_scale
        diff_r = R.from_euler('yzx',-diff_r,degrees=False)
        # diff_p = np.array([-diff_p[1], diff_p[0], diff_p[2]])
        return diff_p, diff_r, width
    
if __name__ == "__main__":
    sigma = Sigma7()
    while True:
        time.sleep(1)
        diff_p, diff_r, width = sigma.get_control()
        print("diff_p:", diff_p)
        print("diff_r:", diff_r.as_euler('yzx',degrees=True))
        print("width:", width)
        print("--------------------")
