import sys

if sys.version_info.major != 3 or sys.version_info.minor < 9:
    print(f"The current python version is: \n {sys.version} \nMinimum requirement: 3.9 !")
    raise RuntimeError("Python Version Error!")



