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
        dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
    
    mean_error = 0
    for i in range(len(objp_list)):
        imgpoints2, _ = cv2.projectPoints(objp_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(corners_list[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "Baseline total error: {}".format(mean_error/len(objp_list)) )

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

    return np.array(residuals) 

def estimate_homography(corners):
    M_list = np.ones((9*6,3), np.float32)
    M_list[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2) # could be backwards 
    zeros = np.zeros((3))
    L_Mat = np.zeros((2*len(corners), 9))
    for i in range(len(corners)):
        # Create sets of L 
        row1 = np.concatenate([M_list[i], zeros, -corners[i][0,0]*M_list[i]], dtype=np.float32)
        row2 = np.concatenate([zeros, M_list[i], -corners[i][0,1]*M_list[i]], dtype=np.float32)
        L = np.vstack((row1,row2))
        L_Mat[(i*2):(i*2+2), :] = L
    
    # print(np.round(L_Mat))
    
    U,S,Vh = np.linalg.svd(L_Mat)
    v_k = np.argmin(S)
    h = Vh[v_k, :] # not a very good 
    # print(f"Before OPTIM:\n {np.reshape(h, (3,3))}")
    """ Use non-linear optim to find better H. (projection of [u,v] onto [X Y] via H)"""
    result = scipy.optimize.least_squares(homography_helper, h, method='lm', args=(corners, M_list))
    # print(result)
    # print(f"After OPTIM:\n {np.reshape(result['x'], (3,3))}")
    return np.reshape(result['x'], (3,3))

def create_v_mat(h, row_1, row_2):
    hi = np.reshape(h[row_1, :], (3,1))
    hj = np.reshape(h[row_2, :], (3,1))

    v = np.concatenate([ hi[0]*hj[1], 
                    hi[0]*hj[1]+hi[1]*hj[0],
                    hi[1]*hj[1],
                    hi[0]*hj[2]+hi[2]*hj[0],
                    hi[1]*hj[2]+hi[2]*hj[1],
                    hi[1]*hj[1]])

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

def get_params_from_b(b):
    param_dict = dict()
    

    param_dict["v0"] = (b[1]*b[3]-b[0]*b[4])/(b[0]*b[2]-(b[1]**2))
    param_dict["lambda"] = b[5] - ((b[3])**2 + param_dict["v0"]*(b[1]*b[3]-b[0]*b[4]))/b[0]
    param_dict["alpha"] = np.sqrt(param_dict["lambda"]/b[0])
    param_dict["Beta"] = np.sqrt(param_dict["lambda"]*b[0]/(b[0]*b[2]-(b[1]**2)))
    param_dict["gamma"] = -b[1]*(param_dict["alpha"]**2)*param_dict["Beta"]/param_dict["lambda"]
    param_dict["u0"] = param_dict["gamma"]*param_dict["v0"]/param_dict["Beta"]-b[3]*(param_dict["alpha"]**2)/param_dict["lambda"]

    param_dict2 = dict()

    w = b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2)
    d = b[0]*b[2]-(b[1]**2)

    param_dict2["alpha"] = np.sqrt(w/(d*b[0]))
    param_dict2["Beta"] = np.sqrt((w/(d**2))*b[0])
    param_dict2["gamma"] = np.sqrt((w/((d**2))*b[0]))*b[1]
    param_dict2["u0"] = (b[1]*b[4]-b[2]*b[3])/d
    param_dict2["v0"] = (b[1]*b[3]-b[0]*b[4])/d

    print(param_dict)
    print(param_dict2)

    return param_dict
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
    for corners in corners_list:
        h = estimate_homography(corners)
        h_list.append(h)

   
    """Use H estim to solve for b problem using 7-9"""

    b = solve_for_b(h_list)

    """Use appendix B to find closed form K from b"""

    parameters = get_params_from_b(b)

    """Estimate [R | T] from A?"""
    """Non-linear solver to increase accuracy of K"""
    """Compute Radial Distortion via non-linear optim..."""
    """Calculate reprojection error for report"""
    """Undistort and Write images for figures in report"""

    pass

def main():
    # calc_intrinsics_baseline()
    calc_intrinsics()


if __name__ == "__main__":
    main()