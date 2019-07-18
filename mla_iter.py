import subprocess as s

import numpy as np

if __name__ == '__main__':

    for i in np.arange(0,5 ):
        print("Processing PyXover series at external iteration",i)
        s.call(["python3", "launch_test.py", "0", "1301", "1", str(i)])
        s.call(["python3", "launch_test.py", "0", "9", "2", str(i)])
        s.call(["python3", "launch_test.py", "0", "9", "3", str(i)])