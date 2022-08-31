import sys
import os

fname = sys.argv[1]
linefirst = int(sys.argv[2])
linelast = int(sys.argv[3])
config = sys.argv[4]

with open(fname, "r") as fi:
    for i, line in enumerate(fi):
        if (i < linefirst) or (i > linelast):
            continue
        t1, t2, t3, t4 = line.rstrip().split(" ")
        command = f"/usr/bin/python3 -m xcell.cls.cov {config} {t1} {t2} {t3} {t4}"
        print(command, flush=True)
        os.system(command)

