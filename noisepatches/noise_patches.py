import os
import glob
import numpy as np
import cv2
import random
# import argparse
# import yaml



# parser = argparse.ArgumentParser(description='create a dataset')
# parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
# parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
# parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
# parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
# opt = parser.parse_args()

# # define input and target directories
# with open('./preprocess/paths.yml', 'r') as stream:
#     PATHS = yaml.load(stream)



def jpeg_compress(img):
  ### JPEG compression test
  # compression = 60
  compression = random.randint(40, 60)
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression] #encoding parameters
  # encode
  is_success, encimg = cv2.imencode('.jpg', img, encode_param) 
  # decode
  noise_img = cv2.imdecode(encimg, 1)
  ### /JPEG compression test
  #noise_img = img.copy()
  return noise_img

#note: cv2 is bgr, not rgb -> image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def noise_patch(rgb_img, patch_size, max_var, min_mean):
    #img = rgb_img.convert('L')
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    #rgb_img = np.array(rgb_img)
    #img = np.array(img)

    w, h = img.shape
    #print(w, h, img.min(), img.max())
    collect_patchs = []

    for i in range(0, w - patch_size, patch_size):
        for j in range(0, h - patch_size, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            var_global = patch.var()
            mean_global = patch.mean()
            #cv2_imshow(patch)
            #print("var_global, mean_global:", var_global, mean_global)
            #print(np.mean(abs(patch - patch.mean())**2))
            if var_global < max_var and mean_global > min_mean:
                rgb_patch = rgb_img[i:i + patch_size, j:j + patch_size, :]
                collect_patchs.append(rgb_patch)

    return collect_patchs


if __name__ == '__main__':

    images_path = "../BasicSR/datasets/DIV2K/HR"
    noise_patches_path = "../noise_patches"
    mode = 'normal' #'normal' #'jpeg'
    noise_type = 'artifact' # 'smartphone' # 'other'

    # https://competitions.codalab.org/competitions/22220
    if noise_type == 'artifact':
        patch_size = 256 # original: 256 #. 100?
        max_var = 20 # original: 20
        min_mean = 0 # original: 0 or 50
    elif noise_type == 'smartphone':
        patch_size = 256
        max_var = 20
        min_mean = 50

    assert not os.path.exists(noise_patches_path)
    os.mkdir(noise_patches_path)

    #TODO: check any image extension
    img_paths = sorted(glob.glob(os.path.join(images_path, '*.png')))
    cnt = 0
    for path in img_paths:
        img_name = os.path.splitext(os.path.basename(path))[0]
        print('Processing: {}'.format(img_name))
        img = cv2.imread(path)
        if mode == 'jpeg':
            img = jpeg_compress(img)

        patches = noise_patch(img, patch_size, max_var, min_mean)

        for idx, patch in enumerate(patches):
            save_path = os.path.join(noise_patches_path, '{}_{:03}.png'.format(img_name, idx))
            cnt += 1
            print('Extracted patch {} as {}'.format(cnt, save_path))
            # print('collect:', cnt)
            cv2.imwrite(save_path, patch)
            # cv2_imshow(patch)
