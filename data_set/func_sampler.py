import numpy as np
import math

# Period = 40 (6.28/48=0,1308)
print('sin_func')
for x in np.arange(0.0, 300, 0.1308):
    print(math.sin(x))

# print('cos_func')
# for x in np.arange(0.0, 300, 0.1308):
#     print(math.cos(x))

