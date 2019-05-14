
# coding: utf-8

# In[1]:


import numpy as np
import PIL.Image
from PIL import Image
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from scipy.ndimage import rank_filter
from scipy.stats import norm
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from scipy.spatial import distance
import math
import copy
from scipy.misc import imsave


# In[2]:


#All Function definitions are placed in this cell

def harris(im, sigma, thresh=None, radius=None):
    """
    Harris corner detector

    Parameters
    ----------
    im: numpy.ndarray
        Image to be processed
    sigma: float
        Standard deviation of smoothing Gaussian
    thresh: float (optional)
    radius: float (optional)
        Radius of region considered in non-maximal suppression

    Returns
    -------
    cim: numpy.ndarray
        Binary image marking corners
    r: numpy.ndarray
        Row coordinates of corner points. Returned only if none of `thresh` and
        `radius` are None.
    c: numpy.ndarray
        Column coordinates of corner points. Returned only if none of `thresh`
        and `radius` are None.
    """
    if im.ndim == 3:
        im = rgb2gray(im)

    dx = np.tile([[-1, 0, 1]], [3, 1])
    dy = dx.T

    Ix = convolve2d(im, dx, 'same')
    Iy = convolve2d(im, dy, 'same')

    f_wid = np.round(3 * np.floor(sigma))
    G = norm.pdf(np.arange(-f_wid, f_wid + 1),
                 loc=0, scale=sigma).reshape(-1, 1)
    G = G.T * G
    G /= G.sum()

    Ix2 = convolve2d(Ix ** 2, G, 'same')
    Iy2 = convolve2d(Iy ** 2, G, 'same')
    Ixy = convolve2d(Ix * Iy, G, 'same')

    cim = (Ix2 * Iy2 - Ixy ** 2) / (Ix2 + Iy2 + 1e-12)

#     if thresh is None or radius is None:
#         return cim
    if thresh and thresh >= 0:
#         size = int(2 * radius + 1)
#         mx = rank_filter(cim, -1, size=size)
#         cim = (cim == mx) & (cim > thresh)
        cim = cim * 255 / cim.max()
        thresh_indices = (cim < thresh)
        cim[thresh_indices] = 0
#         r, c = cim.nonzero()
        return cim
#         return cim, r, c


def gen_dgauss(sigma):
    """
    Generates the horizontally and vertically differentiated Gaussian filter

    Parameters
    ----------
    sigma: float
        Standard deviation of the Gaussian distribution

    Returns
    -------
    Gx: numpy.ndarray
        First degree derivative of the Gaussian filter across rows
    Gy: numpy.ndarray
        First degree derivative of the Gaussian filter across columns
    """
    f_wid = 4 * np.floor(sigma)
    G = norm.pdf(np.arange(-f_wid, f_wid + 1),
                 loc=0, scale=sigma).reshape(-1, 1)
    G = G.T * G
    Gx, Gy = np.gradient(G)

    Gx = Gx * 2 / np.abs(Gx).sum()
    Gy = Gy * 2 / np.abs(Gy).sum()

    return Gx, Gy


def find_sift(I, circles, enlarge_factor=1.5):
    """
    Compute non-rotation-invariant SITF descriptors of a set of circles

    Parameters
    ----------
    I: numpy.ndarray
        Image
    circles: numpy.ndarray
        An array of shape `(ncircles, 3)` where ncircles is the number of
        circles, and each circle is defined by (x, y, r), where r is the radius
        of the cirlce
    enlarge_factor: float
        Factor which indicates by how much to enlarge the radius of the circle
        before computing the descriptor (a factor of 1.5 or large is usually
        necessary for best performance)

    Returns
    -------
    sift_arr: numpy.ndarray
        Array of SIFT descriptors of shape `(ncircles, 128)`
    """
#     assert circles.ndim == 2 and circles.shape[1] == 3, \
#         'Use circles array (keypoints array) of correct shape'
    I = I.astype(np.float64)
    if I.ndim == 3:
        I = rgb2gray(I)

    NUM_ANGLES = 8
    NUM_BINS = 4
    NUM_SAMPLES = NUM_BINS * NUM_BINS
    ALPHA = 9
    SIGMA_EDGE = 1

    ANGLE_STEP = 2 * np.pi / NUM_ANGLES
    angles = np.arange(0, 2 * np.pi, ANGLE_STEP)

    height, width = I.shape[:2]
    num_pts = circles.shape[0]

    sift_arr = np.zeros((num_pts, NUM_SAMPLES * NUM_ANGLES))

    Gx, Gy = gen_dgauss(SIGMA_EDGE)

    Ix = convolve2d(I, Gx, 'same')
    Iy = convolve2d(I, Gy, 'same')
    I_mag = np.sqrt(Ix ** 2 + Iy ** 2)
    I_theta = np.arctan2(Ix, Iy + 1e-12)

    interval = np.arange(-1 + 1/NUM_BINS, 1 + 1/NUM_BINS, 2/NUM_BINS)
    gridx, gridy = np.meshgrid(interval, interval)
    gridx = gridx.reshape((1, -1))
    gridy = gridy.reshape((1, -1))

    I_orientation = np.zeros((height, width, NUM_ANGLES))

    for i in range(NUM_ANGLES):
        tmp = np.cos(I_theta - angles[i]) ** ALPHA
        tmp = tmp * (tmp > 0)

        I_orientation[:, :, i] = tmp * I_mag

    for i in range(num_pts):
        cy, cx = circles[i, :2]
#         r = 32
        r = circles[i, 2]

        gridx_t = gridx * r + cx
        gridy_t = gridy * r + cy
        grid_res = 2.0 / NUM_BINS * r

        x_lo = np.floor(np.max([cx - r - grid_res / 2, 0])).astype(np.int32)
        x_hi = np.ceil(np.min([cx + r + grid_res / 2, width])).astype(np.int32)
        y_lo = np.floor(np.max([cy - r - grid_res / 2, 0])).astype(np.int32)
        y_hi = np.ceil(
            np.min([cy + r + grid_res / 2, height])).astype(np.int32)

        grid_px, grid_py = np.meshgrid(
            np.arange(x_lo, x_hi, 1),
            np.arange(y_lo, y_hi, 1))
        grid_px = grid_px.reshape((-1, 1))
        grid_py = grid_py.reshape((-1, 1))

        dist_px = np.abs(grid_px - gridx_t)
        dist_py = np.abs(grid_py - gridy_t)

        weight_x = dist_px / (grid_res + 1e-12)
        weight_x = (1 - weight_x) * (weight_x <= 1)
        weight_y = dist_py / (grid_res + 1e-12)
        weight_y = (1 - weight_y) * (weight_y <= 1)
        weights = weight_x * weight_y

        curr_sift = np.zeros((NUM_ANGLES, NUM_SAMPLES))
        for j in range(NUM_ANGLES):
            tmp = I_orientation[y_lo:y_hi, x_lo:x_hi, j].reshape((-1, 1))
            curr_sift[j, :] = (tmp * weights).sum(axis=0)
        sift_arr[i, :] = curr_sift.flatten()

    tmp = np.sqrt(np.sum(sift_arr ** 2, axis=-1))
    if np.sum(tmp > 1) > 0:
        sift_arr_norm = sift_arr[tmp > 1, :]
        sift_arr_norm /= tmp[tmp > 1].reshape(-1, 1)

        sift_arr_norm = np.clip(sift_arr_norm, sift_arr_norm.min(), 0.2)

        sift_arr_norm /= np.sqrt(
            np.sum(sift_arr_norm ** 2, axis=-1, keepdims=True))

        sift_arr[tmp > 1, :] = sift_arr_norm

    return sift_arr


def custom_sift(I, important_points, neighborhood):
    result = []
    i1_limit = I.shape[0]
    i2_limit = I.shape[1]
    for point in important_points:
        i = point[0]
        j = point[1]
        current_sift = []
        start = i - neighborhood
        end = i + neighborhood
        for i1 in range(start, end + 1):
            for i2 in range(start, end + 1):
                if 0<= i1 < i1_limit and 0<= i2 < i2_limit:
                    current_sift.append(I[i1, i2])
                else:
                    current_sift.append(0)
        result.append(current_sift)
    return np.array(result)


def dist2(x, c):
    """
    Calculates squared distance between two sets of points.

    Parameters
    ----------
    x: numpy.ndarray
        Data of shape `(ndata, dimx)`
    c: numpy.ndarray
        Centers of shape `(ncenters, dimc)`

    Returns
    -------
    n2: numpy.ndarray
        Squared distances between each pair of data from x and c, of shape
        `(ndata, ncenters)`
    """
    assert x.shape[1] == c.shape[1],         'Data dimension does not match dimension of centers'

    x = np.expand_dims(x, axis=0)  # new shape will be `(1, ndata, dimx)`
    c = np.expand_dims(c, axis=1)  # new shape will be `(ncenters, 1, dimc)`

    # We will now use broadcasting to easily calculate pairwise distances
    n2 = np.sum((x - c) ** 2, axis=-1)

    return n2


def save_img(array, filename):
    """
    Saves given numpy array to ./images folder with given filename
    """
    img_new = Image.fromarray(array)
    img_new = img_new.convert("RGB")
    fp = open("./images/" + filename, "wb")
    img_new.save(fp)
    fp.close()
    return None

def save_img_c(array, filename):
    """
    Saves given numpy array to ./images folder with given filename
    """
    imsave("./images/"+ filename, array)
#     img_new = Image.fromarray(array)
# #     img_new = img_new.convert("RGB")
#     fp = open("./images/" + filename, "wb")
#     img_new.save(fp)
#     fp.close()
#     return None


def find_homography(points_1, points_2):
    #http://ros-developer.com/2017/12/26/finding-homography-matrix-using-singular-value-decomposition-and-ransac-in-opencv-and-matlab/
    x1, y1 = points_1[0][0], points_1[0][1]
    x2, y2 = points_1[1][0], points_1[1][1]
    x3, y3 = points_1[2][0], points_1[2][1]
    x4, y4 = points_1[3][0], points_1[3][1]
    
    xp1, yp1 = points_2[0][0], points_2[0][1]
    xp2, yp2 = points_2[1][0], points_2[1][1]
    xp3, yp3 = points_2[2][0], points_2[2][1]
    xp4, yp4 = points_2[3][0], points_2[3][1]
    
    A = [[-x1, -y1, -1,  0,   0,   0, x1*xp1, y1*xp1, xp1],
         [0  ,  0,   0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
         [-x2, -y2, -1,   0,   0,  0, x2*xp2, y2*xp2, xp2],
         [  0,   0,  0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
         [-x3, -y3, -1,   0,   0,  0, x3*xp3, y3*xp3, xp3],
         [  0,   0,  0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
         [-x4, -y4, -1,   0,   0,  0, x4*xp4, y4*xp4, xp4],
         [  0,   0,  0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]]
    
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    v = np.transpose(vh)
    h = v[:, -1]
    return h.reshape(3,3)

def get_point(h, point):
    t1 = np.array([point[0], point[1], 1])
    t1 = t1.reshape(3,1)
    t1p = np.dot(h, t1)
    t1p = t1p / t1p[-1][0]
    return t1p[:t1p.shape[0] -1, :]

def run_ransac(py_sift_distance, threshold):
#     g_residue = float('inf')
    max_inliers = 0
    iterations = 2000
    average_inliers = 0
    g_h = np.zeros((3,3), dtype= float)
    for i in range(iterations):
        picked_index = np.random.choice(len(py_sift_distance), 4, replace=False)
        p1 = [ py_sift_distance[index][1] for index in picked_index]
        p2 = [ py_sift_distance[index][2] for index in picked_index]
        tmp_homography = find_homography(p1, p2)
        average_residue = 0
        inlier = 0
        for p in py_sift_distance:
            p2_p = get_point(tmp_homography, p[1])
#             print(p2_p)
            residue = distance.euclidean(p[2], p2_p)
            if residue < threshold:
                inlier += 1
                average_residue += residue
        # print("inliers :", inlier, max_inliers)
        average_inliers += inlier
        
        
        if max_inliers < inlier:
            print("Homography updating")
            print("old:", average_residue)
            max_inliers = inlier
            
            g_h = tmp_homography
    print("average_inliers:", average_inliers / iterations)
    return g_h
        


# In[3]:


#https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
I1 = np.asarray(PIL.Image.open('./hw2_data/uttower_left.JPG'))
I2 = np.asarray(PIL.Image.open('./hw2_data/uttower_right.JPG'))

#Setting datatype as float
I1 = I1.astype(float)
I2 = I2.astype(float)

I1_c = I1.copy()
I2_c = I2.copy()
#Converting into greyscale
I1 = rgb2gray(I1)
I2 = rgb2gray(I2)


# In[4]:


#Getting response from Harris Detector
response1 = harris(I1, 1.1, 80, 3)
response2 = harris(I2, 1.1, 80, 3)

save_img(response1, "response1.png")
save_img(response2, "response2.png")


# In[5]:


#http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.peak_local_max
#Using minimum supression on responce array to get important points
importantPoints1 = peak_local_max(response1, min_distance=5, indices=True)
importantPoints2 = peak_local_max(response2, min_distance=5, indices=True)

print("Dimensions of left and right image important points:", importantPoints1.shape, importantPoints2.shape)

#Saving Important points numpy array as image
visualize_ip1 = np.zeros((I1.shape[0], I1.shape[1]), dtype=int)
visualize_ip2 = np.zeros((I2.shape[0], I2.shape[1]), dtype=int)

for point1, point2 in zip(importantPoints1, importantPoints2):
    visualize_ip1[point1[0]][point1[1]] = 255
    visualize_ip2[point2[0]][point2[1]] = 255
    
save_img(visualize_ip1, "visualize_ip1.png")
save_img(visualize_ip2, "visualize_ip2.png")


# In[6]:


#Creating sift descriptors
radius_sift = 32
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html

sift_list1 = find_sift(I1, np.concatenate((importantPoints1, np.ones((importantPoints1.shape[0], 1), dtype = int) * radius_sift), axis=1))
sift_list2 = find_sift(I2, np.concatenate((importantPoints2, np.ones((importantPoints2.shape[0], 1), dtype = int) * radius_sift), axis=1))

print("Dimensions of sift_list 1 and 2:",sift_list1.shape, sift_list2.shape)


# In[7]:


#Create distance matrix and find best match
match_count = 40
sift_distance_matrix = cdist(sift_list1, sift_list2)
py_sift_distance = []
sety = set()
for i in range(sift_distance_matrix.shape[0]):
    for j in range(sift_distance_matrix.shape[1]):
        py_sift_distance.append((sift_distance_matrix[i,j], importantPoints1[i], importantPoints2[j]))

py_sift_distance = sorted(py_sift_distance, key = lambda x: x[0])


# In[8]:


matches1_img = np.zeros(I1.shape, dtype = float)
matches2_img = np.zeros(I2.shape, dtype = float)
for i, t in enumerate(py_sift_distance[:match_count]):
    matches1_img[t[1][0]][t[1][1]] = 255
    matches2_img[t[2][0]][t[2][1]] = 255
    
save_img(matches1_img, 'matches1.png')
save_img(matches2_img, 'matches2.png')


# In[9]:


homography = run_ransac(py_sift_distance[:match_count], 2)


# In[10]:


homography


# In[11]:


get_point(h= homography, point=(562, 782))


# In[12]:


py_sift_distance[34]


# In[13]:


print(I1.shape, I2.shape)


# In[14]:


def place_image(image, canvas, transform, h):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if transform:
                p = get_point(h, (i, j))
#                 print(i, j, p)
                try:
                    canvas [math.ceil(p[0][0]) + 200] [math.ceil(p[1][0] + 1200)] = image[i][j]
                    canvas [math.floor(p[0][0])+ 200] [math.floor(p[1][0] + 1200)] = image[i][j]
                except:
#                     print("original",p[0][0] + 200, p[1][0] +1200)
                    p[0][0] = (math.ceil(p[0][0]) + 200) % canvas.shape[0]
                    p[1][0] = (math.ceil(p[1][0] + 1200)) % canvas.shape[1]
#                     print("modified",p)
                    canvas [(math.ceil(p[0][0]) + 200)% canvas.shape[0]] [(math.ceil(p[1][0] + 1200)) % canvas.shape[1]] = image[i][j]
                    canvas [(math.floor(p[0][0])+ 200)% canvas.shape[0]] [(math.floor(p[1][0] + 1200)) % canvas.shape[1]] = image[i][j]
            else:
                if canvas[i + 200][j + 1200].all() == 0:
                    canvas [i + 200] [j + 1200] = image[i][j]
                else:
                    canvas [i+ 200] [j + 1200] += image[i][j]
                    canvas [i + 200] [j + 1200] /= 2 
            
    return canvas


# In[15]:


I1.shape


# In[16]:



#creating huge canvas for stiched image
I3 = np.zeros((1000, 3000), dtype = float)
c1 = place_image(I1, I3, True, homography)
save_img(c1, 'c1.png')
c2 = place_image(I2, c1, False, homography)
save_img(c2, 'c2.png')


# In[17]:


I3_c = np.zeros((1000, 3000, 3), dtype = float)
c1_c = place_image(I1_c, I3_c, True, homography)
save_img_c(c1_c, 'c1_c.png')
c2_c = place_image(I2_c, c1_c, False, homography)
save_img_c(c2_c, 'c2_c.png')


# In[18]:


# #Saving best homography
# np.savetxt('best_homo.out', homography, delimiter=',')

