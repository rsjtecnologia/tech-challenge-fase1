
import sys
with open("tests/test_cnn.py", "w") as f:
    f.write(open(sys.argv[1]).read())
print("done")
