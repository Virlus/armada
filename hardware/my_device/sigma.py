import numpy as np
import time
from scipy.spatial.transform import Rotation as R

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sigma_sdk"))
import sigma7

class Sigma7:
    def __init__(self, pos_scale=5, width_scale=1000,num_robot=1) -> None:
        self.pos_scale = pos_scale
        self.width_scale = width_scale
        self.start_sigma()
        init_p, init_r, _ = self.read_state()
        self.num_robot = num_robot
        print("=============num_robot", num_robot)
        if num_robot == 1:
            self.init_p = init_p
            self.init_r = init_r
        else:
            self.init_p = []
            self.init_r = []
            self._prev_p = []
            self._prev_r = []
            for i in range(num_robot):
                self.init_p.append(init_p)
                self.init_r.append(init_r)
                self._prev_p.append(init_p)
                self._prev_r.append(init_r)

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
        pos = np.array([px, py,pz])
        rot = np.array([-oa, ob, -og])
        return pos, rot, pg
    
    def get_control(self,rbt_id=0):
        """
        get the difference between cur_p and init_p,cur_r and init_r ,and finally return them
        """
        rbt_id=int(rbt_id)
        curr_p, curr_r, pg = self.read_state()
        if self.num_robot == 1:
            diff_p = curr_p - self.init_p
            diff_r = curr_r - self.init_r
            diff_p = diff_p * self.pos_scale
            width = pg / -0.027 * self.width_scale
            diff_r = R.from_euler('xyz', diff_r,degrees=False)
        else:
            diff_p = curr_p - self.init_p[rbt_id]
            diff_r = curr_r - self.init_r[rbt_id]
            diff_p = diff_p * self.pos_scale
            width = pg / -0.027 * self.width_scale
            diff_r = R.from_euler('xyz', diff_r, degrees=False)
        return diff_p, diff_r, width
    
    def detach(self,rbt_id=0):
        rbt_id = int(rbt_id)
        # raise RuntimeError("not implemented")
        prev_p, prev_r, _ = self.read_state()
        if self.num_robot == 1:
            self._prev_p = prev_p
            self._prev_r = prev_r
            print("------------detach-------------")
        else:
            self._prev_p[rbt_id] = prev_p
            self._prev_r[rbt_id] = prev_r

    def resume(self,rbt_id=0):
        rbt_id = int(rbt_id)
        # raise RuntimeError("not implemented")
        curr_p, curr_r, _ = self.read_state()
        if self.num_robot == 1:
            self.init_p = self.init_p + curr_p - self._prev_p  #renew init_p by adding bias during detachment, which is curr_p - self._prev_p
            self.init_r = self.init_r + curr_r - self._prev_r
            print("------------resume-------------")
        else:
            self.init_p[rbt_id] = self.init_p[rbt_id] + curr_p - self._prev_p[rbt_id]  # renew init_p by adding bias during detachment, which is curr_p - self._prev_p
            self.init_r[rbt_id] = self.init_r[rbt_id] + curr_r - self._prev_r[rbt_id]

    def reset(self,rbt_id = 0):
        rbt_id = int(rbt_id)
        # raise RuntimeError("not implemented")
        """
        read current state of sigma and save as init_p and init_r
        """
        if self.num_robot == 1:
            self.init_p, self.init_r, _ = self.read_state()
            print("------------reset -------------")
        else:
            self.init_p[rbt_id], self.init_r[rbt_id], _ = self.read_state()

    def transform_from_robot(self, translate, rotation,rbt_id=0):
        rbt_id = int(rbt_id)
        """
        update init_p and init_r with dp and dr,which correspond to initial pose of robot
        """
        if self.num_robot == 1:
            self.init_p -= translate / self.pos_scale
            self.init_r -= rotation.as_euler('xyz', degrees=False)
            print("------------transform ------------")
        else:
            self.init_p[rbt_id] -= translate / self.pos_scale
            self.init_r[rbt_id] -= rotation.as_euler('xyz', degrees=False)
    
if __name__ == "__main__":
    num_robot = 1
    sigma = Sigma7(num_robot=num_robot)
    while True:
        time.sleep(1)
        diff_p, diff_r, width = sigma.get_control()
        print("diff_p:", diff_p)
        print("diff_r:", diff_r.as_euler('yzx',degrees=True))
        print("width:", width)
        print("--------------------")
