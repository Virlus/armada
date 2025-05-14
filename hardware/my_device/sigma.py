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
        init_p, init_r, _ = self.read_state()
        self.init_p = init_p
        self.init_r = init_r

    def start_sigma(self):
        sigma7.drdOpen()
        sigma7.drdAutoInit()
        print('starting sigma')
        sigma7.drdStart()
        sigma7.drdRegulatePos(on = False)
        sigma7.drdRegulateRot(on = False)
        sigma7.drdRegulateGrip(on = False)
        print('sigma ready')

    def read_state(self):
        sig, px, py, pz, oa, ob, og, pg, matrix = sigma7.drdGetPositionAndOrientation()
        pos = np.array([-px, -py,pz])
        rot = np.array([oa, -ob, -og])
        return pos, rot, pg
    
    def get_control(self):
        curr_p, curr_r, pg = self.read_state()
        diff_p = curr_p - self.init_p
        diff_r = curr_r - self.init_r
        diff_p = diff_p * self.pos_scale
        width = pg / -0.027 * self.width_scale
        diff_r = R.from_euler('xyz', diff_r,degrees=False)
        return diff_p, diff_r, width
    
    def detach(self):
        prev_p, prev_r, _ = self.read_state()
        self._prev_p = prev_p
        self._prev_r = prev_r

    def resume(self):
        curr_p, curr_r, _ = self.read_state()
        self.init_p = self.init_p + curr_p - self._prev_p
        self.init_r = self.init_r + curr_r - self._prev_r

    def reset(self):
        self.init_p, self.init_r, _ = self.read_state()

    def transform_from_robot(self, translate, rotation):
        self.init_p -= translate / self.pos_scale
        self.init_r -= rotation.as_euler('xyz', degrees=False)
    
if __name__ == "__main__":
    sigma = Sigma7()
    while True:
        time.sleep(1)
        diff_p, diff_r, width = sigma.get_control()
        print("diff_p:", diff_p)
        print("diff_r:", diff_r.as_euler('yzx',degrees=True))
        print("width:", width)
        print("--------------------")
