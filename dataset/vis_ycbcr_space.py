import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    fig = plt.figure(dpi=400)
    r = np.random.rand(1, 100000)
    g = np.random.rand(1, 100000)
    b = np.random.rand(1, 100000)
    delta = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta

    cb_abs = np.abs(cb)
    cr_abs = np.abs(cr)
    res = y - cb_abs - cr_abs
    print(np.min(res))
    print(np.max(res))
    # plt.title("YCbCr Space")
    # ax = Axes3D(fig)
    # ax.scatter(cb, cr, y, s=1)
    # ax.set_xlabel('Cb')
    # ax.set_ylabel('Cr')
    # ax.set_zlabel('Y')
    # plt.show()
