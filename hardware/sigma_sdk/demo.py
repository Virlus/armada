import sigma7
import time
import numpy as np

sigma7.drdOpen()
sigma7.drdAutoInit()
sigma7.drdStart()

sigma7.drdRegulatePos(on = False)
sigma7.drdRegulateRot(on = False)
sigma7.drdRegulateGrip(on = False)

while True:
    print('test: ', sigma7.drdGetPositionAndOrientation())
    time.sleep(0.1)