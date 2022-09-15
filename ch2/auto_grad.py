import torch
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.autograd import grad


def get_function(x1_val=0, x2_val=0, x3_val=0, x4_val=0):
    # variables
    x1 = torch.tensor(x1_val, requires_grad=True, dtype=torch.float32)
    x2 = torch.tensor(x2_val, requires_grad=True, dtype=torch.float32)
    x3 = torch.tensor(x3_val, requires_grad=True, dtype=torch.float32)
    x4 = torch.tensor(x4_val, requires_grad=True, dtype=torch.float32)

    # function
    p1 = x1.pow(3)
    m1 = p1 * x2
    m2 = x3 * x4
    f = m1 + m2
    var = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}
    return f, var

if __name__ == "__main__":
    f, parameters = get_function(2, 4, 3, 5)
    make_dot(f, parameters).render("f_torchviz", format="png")
    print(f.item())
    img = mpimg.imread('f_torchviz.png')
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(img)
    # plt.show()
    df_dx = grad(outputs=f, inputs=parameters.values())
    print(df_dx)
