import numpy as np
import imageio
from matplotlib import pyplot as plt
import cv2

# get every image in a gif
def pre_gif(fname):
    gif = imageio.mimread(fname)
    if (len(gif[0].shape) == 2):
        imgs = [im for im in gif]
    else:
        imgs = [(im[:, :, 0] * 0.3 + im[:, :, 1] * 0.3 + im[:, :, 2] * 0.4) for im in gif]
    return imgs

# correlation function from lab2
def correlation(image, kernel):
    height, width = image.shape
    size = len(kernel)
    s = size // 2
    output = np.float32(np.zeros_like(image))
    # correlation from lab2
    for y in range(s, height - s):
        for x in range(s, width - s):
            output[y, x] = (kernel * image[y - s:y + size - s, x - s:x + size - s]).sum()
    return output

# gaussian filter from lab2
def gaussian(kernel, image):
    height, width = image.shape
    output = np.float32(np.zeros_like(image))
    kSize = kernel.size
    edge = int(kSize / 2)
    # edge remains the same
    output[0:edge, 0:height] = image[0:edge, 0:height]
    output[0:width, 0:edge] = image[0:width, 0:edge]
    output[width - edge:width, 0:height] = image[width - edge:width, 0:height]
    output[0:width, height - edge:height] = image[0:width, height - edge:height]
    # first horizontal, then vertical
    for x in range(edge, height - edge):
        for y in range(edge, width - edge):
            output[y, x] = (kernel * (np.transpose(kernel) * image[y - edge:y + edge + 1, x - edge:x + edge + 1])).sum()
    return output

# scale from lab3
def scale(image):
    scaling = np.array([[0.6, 0, 0], [0, 0.6, 0], [0, 0, 1]])
    A_inv = np.linalg.inv(scaling)
    height, width = np.int16(image.shape)
    height = np.int16(height*0.6)
    width = np.int16(width * 0.6)
    target_nn_img = np.zeros((height,width))

    # nearest neighbor
    for i in range(height):
        for j in range(width):
            target = ([i, j, 1])
            source = np.round(np.dot(A_inv, target))
            m, n, _ = np.int16(source)
            if (0 < m < width and 0 < n < height):
                target_nn_img[i, j] = image[m, n]
    return target_nn_img


# canny edge detection to extra the edge
def canny_edge_detection(img):
    ########################## 1. gaussian blur ##############################
    # size is an odd integer, generate a size x size gaussian kernel
    size = 5
    s = size // 2
    sigma = 1
    x, y = np.mgrid[-s: s + 1, -s: s + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = normal * np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    img = correlation(img, g)[s:-s, s:-s]
    ##########################################################################

    ########################## 2. calculate gradient #########################
    # use sober kernel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    ix = correlation(img, kx)
    iy = correlation(img, ky)

    gradient = np.sqrt(ix ** 2 + iy ** 2)
    gradient = gradient / gradient.max() * 255
    theta = np.arctan2(iy, ix)
    ##########################################################################

    ########################## 3. non-maximum suppression ####################
    non_maximum_img = np.zeros(gradient.shape)
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    def pixel(p1, p2, p3):
        return p1 if p1 == np.max([p1, p2, p3]) else 0

    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            if 0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] <= 180:
                non_maximum_img[i, j] = pixel(gradient[i, j], gradient[i, j - 1], gradient[i, j + 1])
            elif 22.5 <= angle[i, j] < 67.5:
                non_maximum_img[i, j] = pixel(gradient[i, j], gradient[i + 1, j - 1], gradient[i - 1, j + 1])
            elif 67.5 <= angle[i, j] < 112.5:
                non_maximum_img[i, j] = pixel(gradient[i, j], gradient[i + 1, j], gradient[i - 1, j])
            elif 112.5 <= angle[i, j] < 157.5:
                non_maximum_img[i, j] = pixel(gradient[i, j], gradient[i - 1, j - 1], gradient[i + 1, j + 1])

    ##########################################################################

    ########################## 4. double threshold ###########################
    # thresholds are adjustable to achieve better effect
    strong_threshold = non_maximum_img.max() * 0.2
    weak_threshold = non_maximum_img.max() * 0.01
    strong_intensity = 255
    weak_intensity = 50

    double_threshold_img = np.zeros(non_maximum_img.shape)

    double_threshold_img[np.where(non_maximum_img >= strong_threshold)] = strong_intensity
    double_threshold_img[
        np.where((non_maximum_img < strong_threshold) & (non_maximum_img >= weak_threshold))] = weak_intensity
    ##########################################################################

    ########################## 5. edge traching by hysteresis ################
    for i in range(1, double_threshold_img.shape[0] - 1):
        for j in range(1, double_threshold_img.shape[1] - 1):
            if double_threshold_img[i, j] == weak_intensity:
                if (double_threshold_img[i - 1: i + 2, j - 1: j + 2] == strong_threshold).sum() > 0:
                    double_threshold_img[i, j] = strong_intensity
                else:
                    double_threshold_img[i, j] = 0

       
    return double_threshold_img
    ##########################################################################


def optical_flow_Lucas_Kanade(img1, img2, window_size, threshold, x_range, y_range):
    # window_size: To deal with the aperture problem, instead of tracking one pixel,
    #              we choose #window_size pixels to give more constrains of u&v
    # threshold: If the (u or v) is smaller than threshold, which means the change is minor, we discard this change.
    # x_range, y_range: We only care about changes within these ranges.
    ########################## 1. first derivative ############################
    dx = np.array([[-1., 1.], [-1., 1.]])
    dy = np.array([[-1., -1.], [1., 1.]])
    dt = np.array([[1., 1.], [1., 1.]])
    w = np.int(window_size / 2)

    Fx = correlation(img1, dx)
    Fy = correlation(img1, dy)
    Ft = correlation(img2, dt) + correlation(img1, -dt)

    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    ###########################################################################

    ####################   2. Lucas-Kanade ####################################
    for i in x_range:
        for j in y_range:
            fx = Fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            fy = Fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            ft = Ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            if (np.sum(fx)!=0 and np.sum(fy)!=0):

                ft = np.reshape(ft, (ft.shape[0], 1))
                A = np.vstack((fx, fy)).T
                # pseudo inverse
                velocity = np.matmul(np.linalg.pinv(A), ft).flatten()

                if np.abs(velocity[0]) > threshold or np.abs(velocity[1]) > threshold:
                    u[i, j] = velocity[0]
                    v[i, j] = velocity[1]


    return (u, v)
    ###########################################################################

def optical_flow_horn_schunck(img1, img2, threshold, alpha, number_of_iterations):
    dx = np.array([[-1., 1.], [-1., 1.]])
    dy = np.array([[-1., -1.], [1., 1.]])
    dt = np.array([[1., 1.], [1., 1.]])
    weighted = np.array([[1/8, 1/8, 1/8],
                   [1/8,    0, 1/8],
                   [1/8, 1/8, 1/8]])

    u=np.zeros(img1.shape)
    v=np.zeros(img1.shape)

    Fx = correlation(img1, dx)
    Fy = correlation(img1, dy)
    Ft = correlation(img1, dt) + correlation(img2, -dt)

    for _ in range(number_of_iterations):
        u_average = correlation(u, weighted)
        v_average = correlation(v, weighted)

        d = (Fx * u_average + Fy * v_average + Ft) / (alpha**2 + Fx**2 + Fy**2)
        u = u_average - Fx * d
        v = v_average - Fy * d

        u_filter = np.where(abs(u) > threshold, True, False)
        v_filter = np.where(abs(v) > threshold, True, False)
        u = np.where(u_filter & v_filter, u, 0)
        v = np.where(v_filter & v_filter, v, 0)

    return (u,v)

def wrapping (imgs, window_size, threshold, x_range, y_range):
    wrp_img = np.zeros(imgs[0].shape)
    velocity = []
    for img in range(len(imgs)-1):
        u,v = optical_flow_Lucas_Kanade(imgs[img], imgs[img + 1], window_size, threshold, x_range, y_range)
        velocity.append((u,v))

        for i in range(wrp_img.shape[0]):
            for j in range (wrp_img.shape[1]):
                if not (abs(u[i, j]) == 0 or abs(v[i, j]) == 0):
                    wrp_img[i,j] = wrp_img[i,j] + 0.25*(imgs[img])[i][j]
        print('-----------')
    plt.imshow(wrp_img,cmap = 'gray')


def apply_of(imgs, window_size, threshold, x_range, y_range, method=0, alpha = 1, number_of_iterations = 8):

    velocity = []
    for i in range(len(imgs) - 1):
        # get velocity vector
        if method == 0:
            u, v = optical_flow_Lucas_Kanade(imgs[i], imgs[i + 1], window_size, threshold, x_range, y_range)
        else:
            u, v = optical_flow_horn_schunck(imgs[i], imgs[i + 1], threshold, alpha, number_of_iterations)

        print('No.' + np.str(i + 1) + ' image has been processed...  ' + np.str(
            len(imgs) - i - 1) + ' left. Thanks for your patience')
        velocity.append((u, v))


        plt.subplot(2, 2, i + 1)
        plt.axis('off')
        plt.subplots_adjust(left=0.01, right=0.95, bottom=0.01, top=0.95, hspace=0.01, wspace=0.01)
        plt.imshow(imgs[i], cmap='gray')
        

        # draw arrow
        sparse_arrow = np.arange(min(x_range), max(x_range), window_size)
        for i in sparse_arrow:
            for j in sparse_arrow:
                if not (abs(u[i, j]) == 0 or abs(v[i, j]) == 0):
                    plt.arrow(j, i, v[i, j], u[i, j], color='red', head_width=1, head_length=1, width=0.01)
        

    plt.show()


# apply optical flow on a whole image, may take some time
def save_gif(imgs, window_size, threshold,x_range, y_range):
    velocity = []
    for i in range(len(imgs)-1):
        print (i)
        u, v = optical_flow_Lucas_Kanade(imgs[i], imgs[i + 1], window_size, threshold, x_range, y_range)
        velocity.append((u, v))
        sparse_arrow = np.arange(0, imgs[i].shape[1], window_size)
        for m in sparse_arrow:
            for n in sparse_arrow:
                if not (abs(u[m, n]) == 0 or abs(v[m, n]) == 0):

                    end_x = m+np.array(v[m,n]*4).astype(np.int)
                    end_y = n+np.array(u[m,n]*4).astype(np.int)
                    cv2.arrowedLine(imgs[i], (end_y,end_x),(n,m),(255,0,0),1)
    print ('----------------------')
    imageio.mimsave('output.gif', imgs,fps=1)


'''
###############################   LK.1   #####################################
####################    estimated run time: 1 min  ###########################
traffic_imgs = pre_gif("traffic_4_seq.gif")
window_size = 5
threshold = 1
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(traffic_imgs, window_size, threshold,x_range,y_range)
##############################################################################

###############   save the whole sequence as a gif     #######################
##############  estimated run time: 40 min    sorry..  ########################
#save_gif(traffic_imgs, window_size, threshold,x_range,y_range)
##############################################################################

##################    wrapping moving object    ##############################
####################    estimated run time: 1 min  ###########################
#wrapping(traffic_imgs, window_size, threshold,x_range,y_range)
##############################################################################
'''


'''
#### LK.1.2  pre process using gaussian filter to reduce running time ########
####################    estimated run time: 1 min  ###########################
traffic_imgs = pre_gif("traffic_4_seq.gif")
kernel_gaussian = np.array([[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.06]])# 7*7 gaussian filter
gaussian_imgs = [gaussian(kernel_gaussian,im) for im in traffic_imgs]
scale_imgs = [scale(im) for im in gaussian_imgs]
window_size = 7
threshold = 1.2
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(gaussian_imgs, window_size, threshold,x_range,y_range)
##############################################################################
'''

'''
#################  LK.1.3.1 2D rotation optical flow  ########################
####################    estimated run time: 1 min  ###########################
rotation_imgs = pre_gif("2d_rotation.gif")
window_size = 15
threshold = 0.5
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(rotation_imgs, window_size, threshold,x_range,y_range)
##############################################################################
'''

'''
#################  LK.1.3.2 3D rotation optical flow   #######################
####################    estimated run time: 3 mins  ##########################
traffic_imgs = pre_gif("3d_rotation.gif")
window_size = 11
threshold = 4
x_range = np.arange(20,520)
y_range = np.arange(20,520)
apply_of(traffic_imgs, window_size, threshold,x_range,y_range)
##############################################################################
'''

'''
#################  HK.1 traffic moving optical flow default    ###############
####################    estimated run time: 1 mins  ##########################
traffic_imgs = pre_gif("traffic_4_seq.gif")
window_size = 5
threshold = 1
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(traffic_imgs, window_size, threshold,x_range,y_range, method = 1)
##############################################################################
'''

'''
#################  HK.2 traffic moving optical flow alpha = 13    ############
####################    estimated run time: 1 mins  ##########################
traffic_imgs = pre_gif("traffic_4_seq.gif")
window_size = 5
threshold = 1
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(traffic_imgs, window_size, threshold,x_range,y_range, method = 1, alpha = 13)
##############################################################################
'''



#################  HK.3 traffic moving optical flow iterations = 50    ############
####################    estimated run time: 1 mins  ##########################
traffic_imgs = pre_gif("traffic_4_seq.gif")
window_size = 5
threshold = 1
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(traffic_imgs, window_size, threshold,x_range,y_range, method = 1, number_of_iterations = 50)
##############################################################################


'''
#################  HK.4 traffic moving optical flow with gaussian    ############
####################    estimated run time: 1 mins  ##########################
traffic_imgs = pre_gif("traffic_4_seq.gif")
kernel_gaussian = np.array([[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.06]])# 7*7 gaussian filter
traffic_imgs = [gaussian(kernel_gaussian,im) for im in traffic_imgs]
window_size = 5
threshold = 1
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(traffic_imgs, window_size, threshold,x_range,y_range, method = 1, alpha = 13)
##############################################################################
'''

'''
####################  edge detection + optical flow.3   ######################
####################    estimated run time: 1 min  ###########################
imgs = pre_gif("dog.gif")
canny_img = [canny_edge_detection(im) for im in imgs]

#for i in range(len(canny_img)):
#    canny_img[i] = np.where(canny_img[i] > 0 , 0, 255)
#    plt.imshow(canny_img[0],cmap = 'gray')

window_size = 5
threshold = 0
method = 1
x_range = np.arange(10,190)
y_range = np.arange(10,190)
apply_of(canny_img, window_size, threshold,x_range,y_range)
##############################################################################
'''

'''
running_imgs = pre_gif("running.gif")
window_size = 9
threshold = 0
x_range = np.arange(10,160)
y_range = np.arange(10,190)
apply_of(running_imgs, window_size, threshold,x_range,y_range)
'''

