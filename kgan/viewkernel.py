import numpy as np
import cv2


def denorm(x, min_max=(-1.0, 1.0)):
    '''
        Denormalize from [-1,1] range to [0,1]
        formula: xi' = (xi - mu)/sigma
        Example: "out = (x + 1.0) / 2.0" for denorm 
            range (-1,1) to (0,1)
        for use with proper act in Generator output (ie. tanh)
    '''
    print(min_max)
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    # if isinstance(x, torch.Tensor):
    #     return out.clamp(0, 1)
    if isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")


def vis_kernel(k_2):
    # visualization
    #'''
    img = k_2.copy()
    img = denorm(img, min_max=(img.min(), img.max()))
    scale = 16
    newdim=(scale*img.shape[1], scale*img.shape[0]) # W, H
    interpol = cv2.INTER_NEAREST
    img = cv2.resize(img, newdim, interpolation = interpol)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #'''


#path = './results/0006_s003llllllll/0006_s003_kernel_x2.npy'

name = 'cart0700_s185_kernel_x2'
#name = 'cart0701_s093_kernel_x2'
#name = 'cropslice_628_128_128_kernel_x2'
path = './results/{}.npy'.format(name)

#with open(os.path.join(conf.output_dir_path, '%s_kernel_x2.npy' % conf.img_name), 'wb') as f:
with open(path, 'rb') as f:
    kernel = np.load(f)

print(kernel)
print(kernel.shape)
vis_kernel(kernel)
