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
    k = dist[0, :2]
    extrinsics_list = []

    for rvec, tvec in zip(rvecs, tvecs):
        R, __ = cv2.Rodrigues(rvec)
        extr = np.hstack((R, tvec))
        extrinsics_list.append(extr)
    
    calculate_reprojection_error(mtx, extrinsics_list, k, corners_list)

def homography_helper(h, corners, M_List):
    residuals = []
    for corner, M_tilde in zip(corners, M_List):
        corner = np.reshape(corner, (2,1))
        h1 = np.reshape(h[:3], (1,3))
        h2 = np.reshape(h[3:6], (1,3))
        h3 = np.reshape(h[6:], (1,3))
        first_part = 1.0 / np.dot(h3, M_tilde)
        second_part = np.array([np.dot(h1,M_tilde),np.dot(h2, M_tilde)]) 
        x_j = first_part * second_part
        error = np.sum(np.square(corner-x_j))
        
        residuals.extend(error.flatten())

    return np.array(residuals) 

def estimate_homography(corners, M_list):
    corners_copy = copy.deepcopy(corners)
    zeros = np.zeros((3))
    L_Mat = np.zeros((2*len(corners), 9))
    for i in range(len(corners_copy)):
        # Create sets of L 
        row1 = np.concatenate([M_list[i], zeros, -corners_copy[i][0,0]*M_list[i]], dtype=np.float32)
        row2 = np.concatenate([zeros, M_list[i], -corners_copy[i][0,1]*M_list[i]], dtype=np.float32)
        L = np.vstack((row1,row2))
        L_Mat[(i*2):(i*2+2), :] = L
    
    
    U,S,Vh = np.linalg.svd(L_Mat)
    v_k = np.argmin(S)
    h = Vh[v_k, :]
    
    """ Use non-linear optim to find better H. (projection of [u,v] onto [X Y] via H)"""
    result = scipy.optimize.least_squares(homography_helper, h, method='lm', args=(corners_copy, M_list))
    h_optim = result['x']
    return np.reshape(h_optim, (3,3))

def create_v_mat(h, row_1, row_2):
    hi = np.reshape(h[:, row_1], (3,1))
    hj = np.reshape(h[:, row_2], (3,1))
    
    v = np.concatenate([ hi[0]*hj[0], 
                    hi[0]*hj[1]+hi[1]*hj[0],
                    hi[1]*hj[1],
                    hi[2]*hj[0]+hi[0]*hj[2],
                    hi[2]*hj[1]+hi[1]*hj[2],
                    hi[2]*hj[2]])

    return v

def solve_for_b(h_list):
    V_Mat = np.zeros(((2*int(len(h_list))), 6))

    for i in range(0,len(h_list)):
        # pairs of h matrices for this
        v12 = create_v_mat(h_list[i], 0, 1)
        v11 = create_v_mat(h_list[i], 0, 0)
        v22 = create_v_mat(h_list[i], 1, 1)
        V = np.vstack((v12, v11-v22))
        V_Mat[(i*2):(i*2+2), :] = V

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

    param_dict["alpha"] = alpha
    param_dict["beta"] = beta  
    param_dict["gamma"] = gamma
    param_dict["lambda"] = lam 
    param_dict["u0"] = u0      
    param_dict["v0"] = v0      

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
        R_real = np.dot(U, Vh) 
        extrinisc_mat = np.hstack((R,t))
        extriniscs_list.append(extrinisc_mat)
    return extriniscs_list

def calculate_reprojection_error(K_mat, extriniscs_list, k, corners_list):
    mean_error = 0
    M_list2 = np.ones((9*6,4), np.float32)
    M_list2[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)
    M_list2[:, 2] = 0
    mean_error = 0
    for i in range(len(corners_list)):
        for j in range(len(corners_list[i])):
            corner = corners_list[i][j]

            projected = np.dot(extriniscs_list[i], M_list2[j])
            projected /= projected[2]

            # adjust for radial
            r2 = projected[0]**2 + projected[1]**2
            distortion = 1 + k[0] * r2 + k[1] * r2**2
            x_d, y_d = projected[0] * distortion, projected[1] * distortion
            projected_undistored = np.hstack((x_d, y_d, 1))
            projected_undistored = np.dot(K_mat, projected_undistored)
            error = corner - projected_undistored[:2]
            mean_error += cv2.norm(error, cv2.NORM_L2)
    print( "Experimental total error: {}".format(mean_error/len(corners_list)) ) # 39.4 with no distortion estim...


def calc_intrinsics():
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
    print(np.round(K_mat,3))

    """Estimate [R | T] from K and h"""

    extriniscs_list = estimate_extrinsics(K_mat, h_list)

    k = np.array([[0],[0]]) # k is dominated by K and can be set to 0 as an init guess


    """Calculate reprojection error for report"""

    calculate_reprojection_error(K_mat, extriniscs_list, k, corners_list)


    """Undistort and Write images for figures in report"""
    k_undist = np.vstack((k, np.array([[0],[0]])))
    for image, im_name in zip(images_RGB, image_names):
        dst = cv2.undistort(image, K_mat, k_undist, None, None)
        dst_copy = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(dst_copy,(6,9), None)
        corners2 = cv2.cornerSubPix(dst_copy, corners, WINDOW_SIZE, ZERO_ZONE, CRITERIA)
        cv2.drawChessboardCorners(dst, (6,9), corners2, True)
        cv2.imwrite(f"undistored_{im_name}", dst)

def main():
    # calc_intrinsics_baseline()
    calc_intrinsics()


if __name__ == "__main__":
    main()