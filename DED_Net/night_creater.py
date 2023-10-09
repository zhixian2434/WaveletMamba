from hashlib import new
import os
import random
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


def l2_match(x1, x2):
    x1 = x1/255.0
    x2 = x2/255.0
    a = np.mean(x1)-np.mean(x2)
    x2 = x2 + a
    x = x1 - x2
    x = np.power(x, 2)
    return np.mean(x)


def ssim_match(x1,x2):
    return 1-structural_similarity(x1,x2)


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * im + mean_b
    q[q < 0] = 0
    q[q > 255] = 255
    return q


def create_V_patch(image_name):
    num, _ = image_name.split('.')
    out_path = './NightRef_Vpatch/'
    BGR = cv2.imread('./NightRef/'+image_name)
    HSV = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    V0 = cv2.resize(V, (128, 128))
    V1 = cv2.flip(V0, 1)
    V2 = cv2.flip(V0, 0) 
    cv2.imwrite(out_path+num+'_0.jpg', V0)
    cv2.imwrite(out_path+num+'_1.jpg', V1)
    cv2.imwrite(out_path+num+'_2.jpg', V2)
    return


def dc(image_path, refimg_V_patch_list):
    out_path = image_path.replace('mini_train_gth_deleted', 'mini_train_night_deleted_ssimmatch')
    # out_path = out_path.replace('jpg', 'png')
    # out_path = out_path.replace('JPG', 'png')
    #out_path_orgV = image_path.replace('mini_val_gth', 'mini_val_night_orgV')
    #out_path_newV = image_path.replace('mini_val_gth', 'mini_val_night_newV')

    origin_img = cv2.imread(image_path)
    orgimg_HSV = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
    org_H, org_S, org_V = cv2.split(orgimg_HSV)

    org_V_patch = cv2.resize(org_V, (128, 128))

    min_loss = 9999
    min_name = ''
    for refimg_V_patch_name in refimg_V_patch_list:
        cur_no = int(refimg_V_patch_name.split('_')[0])
        if match_count[cur_no] >= 80:  
            continue
        ref_V_patch = cv2.imread('./NightRef_Vpatch/'+refimg_V_patch_name, 0)  
        loss = ssim_match(org_V_patch, ref_V_patch)
        if loss < min_loss:
            min_loss = loss
            min_name = refimg_V_patch_name

    print(image_path + ' match ' + min_name + ', loss=' + str(min_loss))
    match_no = int(min_name.split('_')[0])
    match_count[match_no] = match_count[match_no]+1
    night_V = cv2.imread('./NightRef_Vpatch/' + min_name, 0)
    night_V = cv2.resize(night_V, (256, 256))
    new_V = Guidedfilter(org_V, night_V, 64, 0.0001)
    new_V[new_V < 0] = 0.0
    new_V[new_V > 255.0] = 255.0
    new_V = new_V.astype(np.uint8)
    #cv2.imwrite(out_path_orgV, org_V)
    #cv2.imwrite(out_path_newV, new_V)

    enhancement_result = cv2.merge([org_H, org_S, new_V])
    enhancement_result = cv2.cvtColor(enhancement_result, cv2.COLOR_HSV2BGR)
    cv2.imwrite(out_path, enhancement_result)


if __name__ == '__main__':
    refPath = "NightRef/"
    orgPath = 'mini_train_gth_deleted/'
    ref_list = os.listdir(refPath)
    org_list = os.listdir(orgPath)
    '''
    for ref_name in ref_list:
        create_V_patch(ref_name)
    '''
    ref_V_patch_list = os.listdir('NightRef_Vpatch/')
    match_count = [0 for i in range(333)]
    for org_name in org_list:
        dc(orgPath+org_name, ref_V_patch_list)

    data_name = [match_count]
    colume_name = ['Num']
    chart = pd.DataFrame(columns=colume_name,data=match_count)
    chart.to_csv('./match_count/mini_train_gth_deleted_ssimmatch.csv')
    