import numpy as np
import scipy
import cv2
import os
import copy

import scipy.optimize

WINDOW_SIZE = (11,11)
ZERO_ZONE = (-1, -1)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
np.set_printoptions(suppress=True)

def load_images(im_path: str, num_images: int, flags: int = cv2.IMREAD_GRAYSCALE) -> tuple[list[cv2.Mat], list[str]]:
    images = []
    image_names = []
    filenames = os.listdir(im_path)
    count = 0
    for filename in filenames:
        full_image_path = im_path+filename
        image = cv2.imread(full_image_path, flags=flags)
        image_names.append(filename)
        images.append(image)
        count +=1
        if count == num_images:
            break
    return images, image_names

def calc_intrinsics_baseline():
    images_RGB, im_names = load_images("Calibration_Imgs/", -1, cv2.IMREAD_COLOR_RGB)
    images_GS = []
    im_names_GS = []
    
    corners_list = []
    objp_list = []
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    width, height = images_RGB[0].shape[:2]

    for image, name in zip(images_RGB, im_names):
        success, corners = cv2.findChessboardCorners(image, (6,9), None)
        if success:
            im_copy = copy.deepcopy(image)
            im_copy = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)
            corners = cv2.cornerSubPix(im_copy, corners, WINDOW_SIZE, ZERO_ZONE, CRITERIA)
            corners_list.append(corners)
            objp_list.append(objp)
            images_GS.append(im_copy)
            im_names_GS.append(name)
        else:
            print("Failure to detect corners")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, im_copy.shape[::-1], None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (height,width), 1, (height,width)) # roi is usable area of image
    print("\n")
    print(mtx)
    print("\n")
    print(dist) # k1, k2, p1, p2
    print("\n")
    print(newcameramtx)

    for image, im_name in zip(images_GS, im_names_GS):
        dst = cv2.undistort(image, newcameramtx, dist, None, newcameramtx)
        x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        # cv2.imshow('undistorted', cv2.resize(dst, dsize=(int(dst.shape[1] * .5), int(dst.shape[0]*.5))))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    mean_error = 0
    for i in range(len(objp_list)):
        imgpoints2, _ = cv2.projectPoints(objp_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(corners_list[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print("Baseline total error: {}".format(mean_error/len(objp_list)) ) # 0.0876 for baseline

def homography_helper(h, corners, M_List):
    residuals = []
    for corner, M_tilde in zip(corners, M_List):
        corner = np.reshape(corner, (2,1))
        h1 = np.reshape(h[:3], (1,3))
        h2 = np.reshape(h[3:6], (1,3))
        h3 = np.reshape(h[6:], (1,3))
        # M = np.reshape(M_tilde[0:2], (1,2))
        first_part = 1.0 / np.dot(h3, M_tilde)
        second_part = np.array([np.dot(h1,M_tilde),np.dot(h2, M_tilde)]) # technically backwards from the given eq...
        x_j = first_part * second_part
        error = np.sum(np.square(corner-x_j))
        
        residuals.extend(error.flatten())

    # print(f"Errors: {np.round(np.sum(residuals),4)}")
    return np.array(residuals) 

def estimate_homography(corners, M_list):
    # scale_factor = np.max(np.abs(M_list))
    # M_list /= scale_factor
    # corners_copy = copy.deepcopy(corners) / scale_factor
    corners_copy = copy.deepcopy(corners)
    zeros = np.zeros((3))
    L_Mat = np.zeros((2*len(corners), 9))
    for i in range(len(corners_copy)):
        # Create sets of L 
        row1 = np.concatenate([M_list[i], zeros, -corners_copy[i][0,0]*M_list[i]], dtype=np.float32)
        row2 = np.concatenate([zeros, M_list[i], -corners_copy[i][0,1]*M_list[i]], dtype=np.float32)
        L = np.vstack((row1,row2))
        L_Mat[(i*2):(i*2+2), :] = L
    
    # print(np.round(L_Mat))
    
    U,S,Vh = np.linalg.svd(L_Mat)
    v_k = np.argmin(S)
    h = Vh[v_k, :] # not a very good 
    # h /= h[-1]

    # h_init, __ = cv2.findHomography(M_list, corners_copy, method=0)
    
    """ Use non-linear optim to find better H. (projection of [u,v] onto [X Y] via H)"""
    result = scipy.optimize.least_squares(homography_helper, h, method='lm', args=(corners_copy, M_list))

    # h *= scale_factor
    # h_init *= scale_factor
    # corners_copy *= scale_factor
    # M_list *= scale_factor
    # h_optim = result['x'] * scale_factor
    h_optim = result['x']

    # print(np.round(h_init,6))
    # print("\n")
    # print(np.round(np.reshape(h, (3,3)), 6))
    # print("\n")
    # print(np.round(np.reshape(h_optim,(3,3)),6))

    # print(f"Det on cv2.fh: {np.round(np.linalg.det(h_init),6)}")
    # print(f"Err cv2.fh: {np.round(np.sum(homography_helper(h_init.flatten(), corners_copy, M_list)),6)}")
    # print("\n")
    # print(f"Det Before: {np.round(np.linalg.det(np.reshape(h, (3,3))),6)}")
    # print(f"Err Before: {np.round(np.sum(homography_helper(h, corners_copy, M_list)),6)}")
    # print("\n")
    # print(f"Det After: {np.round(np.linalg.det(np.reshape(h_optim, (3,3))),6)}")
    # print(f"Err After: {np.round(np.sum(homography_helper(h_optim, corners_copy, M_list)),6)}")
    # exit(1)
    return np.reshape(h_optim, (3,3))

def create_v_mat(h, row_1, row_2):
    hi = np.reshape(h[:, row_1], (3,1))
    hj = np.reshape(h[:, row_2], (3,1))

    element_0 = hi[0]*hj[0]
    element_1 = hi[0]*hj[1]+hi[1]*hj[0]
    element_2 = hi[1]*hj[1]
    element_3 = hi[2]*hj[0]+hi[0]*hj[2]
    element_4 = hi[2]*hj[1]+hi[1]*hj[2]
    element_5 = hi[2]*hj[2]
    
    # check math here..
    v = np.concatenate([ hi[0]*hj[0], 
                    hi[0]*hj[1]+hi[1]*hj[0],
                    hi[1]*hj[1],
                    hi[2]*hj[0]+hi[0]*hj[2],
                    hi[2]*hj[1]+hi[1]*hj[2],
                    hi[2]*hj[2]])

    return v

def solve_for_b(h_list):
    #TODO: CHECK 
    V_Mat = np.zeros(((2*int(len(h_list))), 6))

    for i in range(0,len(h_list)):
        # pairs of h matrices for this
        v12 = create_v_mat(h_list[i], 0, 1)
        v11 = create_v_mat(h_list[i], 0, 0)
        v22 = create_v_mat(h_list[i], 1, 1)
        V = np.vstack((v12, v11-v22))
        V_Mat[(i*2):(i*2+2), :] = V

    # print(np.round(V_Mat, 4))
    U,S,Vh = np.linalg.svd(V_Mat)
    v_k = np.argmin(S)
    b = Vh[v_k, :] 
    return b

def make_K_mat(param_mat):
    K = np.zeros((3,3))
    K[0,0] = param_mat["alpha"]
    K[0,1] = param_mat["gamma"]
    K[0,2] = param_mat["u0"]
    K[1,1] = param_mat["beta"]
    K[1,2] = param_mat["v0"]
    K[2,2] = 1
    return K

def get_params_from_b(b):
    param_dict = dict()
    

    v0 = (b[1]*b[3]-b[0]*b[4])/(b[0]*b[2]-(b[1]**2))
    lam = b[5] - ((b[3])**2 + v0*(b[1]*b[3]-b[0]*b[4]))/b[0]
    alpha = np.sqrt(lam/b[0])
    beta = np.sqrt(lam*b[0]/(b[0]*b[2]-(b[1]**2)))
    gamma = -b[1]*(alpha**2)*beta/lam
    u0 = gamma*v0/beta-b[3]*(alpha**2)/lam

    param_dict["alpha"] = alpha         #/ lam
    param_dict["beta"] = beta           #/ lam
    param_dict["gamma"] = gamma         #/ lam
    param_dict["lambda"] = lam          #/ lam
    param_dict["u0"] = u0               #/ lam
    param_dict["v0"] = v0               #/ lam

    param_dict2 = dict()

    w = b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2)
    d = b[0]*b[2]-(b[1]**2)

    param_dict2["alpha"] = np.sqrt(w/(d*b[0]))
    param_dict2["beta"] = np.sqrt((w/(d**2))*b[0])
    param_dict2["gamma"] = np.sqrt((w/((d**2)*b[0])))*b[1] # this is positive VS other one is negative...
    param_dict2["u0"] = (b[1]*b[4]-b[2]*b[3])/d
    param_dict2["v0"] = (b[1]*b[3]-b[0]*b[4])/d

    return param_dict

def estimate_extrinsics(K_mat, h_list):
    extriniscs_list = []

    for h in h_list:
        scalar = 1 / np.linalg.norm(np.linalg.inv(K_mat))
        h1 = np.reshape(h[:, 0], (3,1))
        h2 = np.reshape(h[:, 1], (3,1))
        h3 = np.reshape(h[:, 2], (3,1))

        r1 = scalar * np.dot(np.linalg.inv(K_mat), h1)
        r2 = scalar * np.dot(np.linalg.inv(K_mat), h2)
        r3 = np.linalg.cross(h1.T,h2.T).T
        t =  scalar * np.dot(np.linalg.inv(K_mat), h3)

        R = np.hstack((r1, r2, r3))

        U, S, Vh = np.linalg.svd(R)
        R_real = np.dot(U, Vh) # maybe the transpose of V... Might be useful...
        # print(R)
        # print(R_real)

        extrinisc_mat = np.vstack((np.hstack((R,t)), np.array([0, 0, 0, 1]))) # could be R_real... 
        # print(extrinisc_mat)
        extriniscs_list.append(extrinisc_mat)
    return extriniscs_list

def calc_intrinsics():
    # We know that there is a set of (6,9) grid cell corners that are usable for this calibration
    # Also, each corner is exactly 21.5mm apart 
    # We can get b from sets of v's stack and solved using SVD

    images_RGB, image_names = load_images("Calibration_Imgs/", -1, cv2.IMREAD_COLOR_RGB)
    corners_list = [] # contains sets of 2-D points from diffrent images in the image frame. 

    for i in range(len(images_RGB)):
        __, corners = cv2.findChessboardCorners(images_RGB[i], (6,9), None) 
        corners = cv2.cornerSubPix(cv2.cvtColor(images_RGB[i], cv2.COLOR_BGR2GRAY), corners, WINDOW_SIZE, ZERO_ZONE, CRITERIA)
        corners_list.append(corners)
    
    """Estimate homography via points on the image using DLT equation 2 on page 17"""
    h_list = []
    # for each image, we estimate a homography
    M_list = np.ones((9*6,3), np.float32)
    M_list[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    for corners in corners_list:
        # print("New Estimate: \n")
        h = estimate_homography(corners, M_list)
        h_list.append(h)

   
    """Use H estim to solve for b problem using 7-9"""

    b = solve_for_b(h_list)

    """Use appendix B to find closed form K from b"""

    parameters = get_params_from_b(b)


    K_mat = make_K_mat(parameters)
    print(np.round(K_mat,5))

    """Estimate [R | T] from A?"""

    extriniscs_list = estimate_extrinsics(K_mat, h_list)

    """Non-linear solver to increase accuracy of K"""
        # skip?

    """Compute Radial Distortion via non-linear optim..."""
    k = np.array([[0],[0]])

    """Calculate reprojection error for report"""

    mean_error = 0
    print(extriniscs_list[0])
    a = extriniscs_list[0][0:3,0:3]
    b = np.reshape(extriniscs_list[0][0:3, 3], (3,1))
    print(a)
    print(b)
    for i in range(len(corners_list)):
        imgpoints2, _ = cv2.projectPoints(M_list, extriniscs_list[i][0:3,0:3], np.reshape(extriniscs_list[0][0:3, 3], (3,1)), K_mat, np.array([0,0,0,0]))
        error = cv2.norm(corners_list[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "Experimental total error: {}".format(mean_error/len(corners_list)) ) # 50.42 with no distortion estim...
    """Undistort and Write images for figures in report"""

    # for image, im_name in zip(images_RGB, image_names):
    #     dst = cv2.undistort(image, K_mat, np.array([0,0,0,0]), None, K_mat)
    #     # x, y, w, h = roi
    #     # dst = dst[y:y+h, x:x+w]
    #     cv2.imshow('undistorted', cv2.resize(dst, dsize=(int(dst.shape[1] * .5), int(dst.shape[0]*.5))))
    #     cv2.imshow('regular', cv2.resize(image, dsize=(int(image.shape[1] * .5), int(image.shape[0]*.5))))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    pass

def main():
    calc_intrinsics_baseline()
    # calc_intrinsics()


if __name__ == "__main__":
    main()