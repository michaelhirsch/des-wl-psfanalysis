import numpy as np

def i3_image_get_top_hat(nx_low_res, ny_low_res, sub):

    U = 1.0 / (sub * sub)
    nx = nx_low_res*sub
    ny = ny_low_res*sub
    
    image = np.zeros((nx, ny))

    xc = nx/2-sub/2
    yc = ny/2-sub/2
    i = 0
    while i<sub:
        j = 0
        while j<sub:
            image[xc+i, yc+j] = U
            j += 1
        i += 1

    return image

def i3_image_centered_into(image):

    nx = image.shape[0]
    ny = image.shape[1]

    if nx%2==1:
        i_center = nx/2+1
        j_center = ny/2+1
    else:
        i_center = nx/2
        j_center = ny/2

    result = np.zeros((image.shape[0], image.shape[1]))

    j = 0
    while j < ny:
        i = 0
        while i < nx:
            i_new = i_center + i
            j_new = j_center + j

            if i_new>=nx:
                i_new-=nx
            if j_new>=ny:
                j_new-=ny

            result[j_new,i_new]=image[j][i];
            i += 1
        j += 1

    return result

def i3_image_centered(image):

    return i3_image_centered_into(image)

def i3_image_fourier(image):
    
    return np.fft.fft2(image)
    
def i3_fourier_conv_kernel(N_pix, N_sub, N_pad, image_psf):

    N_all = N_pad + N_pix

    image_pix = i3_image_get_top_hat(N_all, N_all, N_sub)
    image_pix_c = i3_image_centered(image_pix)
    fourier_pix = i3_image_fourier(image_pix_c)
    
    image_psf_c = i3_image_centered(image_psf)
    fourier_psf = i3_image_fourier(image_psf_c)
    
    fourier_ker = fourier_pix * fourier_psf

    return np.real(np.fft.fftshift(np.fft.ifft2(fourier_ker)))


def i3_image_dsample_cut_into(img_src, stamp_size, i_start, j_start, n_sub):

    img_dst = np.zeros((stamp_size, stamp_size))
    
    if n_sub == 1:
        return img_src

    j_dst = 0
    while j_dst < img_dst.shape[1]:
        i_dst = 0
        while i_dst < img_dst.shape[0]:
            i_src = i_dst*n_sub + i_start
            j_src = j_dst*n_sub + j_start
            if (i_src >= img_src.shape[0]) or (j_src >= img_src.shape[1]):
                raise IOError("Bugger. i_src=%d, img_src.shape[0]=%d, i_src=%d, img_src.shape[1]=%d" % (i_src, img_src.shape[0], j_src, img_src.shape[1]))
            img_dst[i_dst, j_dst] = img_src[i_src, j_src]
            i_dst += 1
        j_dst += 1
    
    return img_dst

def i3_downsample(psf, upsampling, stamp_size, padding=0):
    """
    This function is a convenience function around
    'i3_image_dsample_cut_into' The only thing it does is applying
    downsampling to an input function.  There is no 1-to-1 equivalent
    in im3shape; rather this function implements what is done at the
    end of 'i3_sersics_model_image_save_components' in 'i3_image.c',
    the main function for building model images. Here we also consider
    only the case where the image needs to be downsampled rather than
    making the case distinction at the end of
    'i3_sersics_model_image_save_components' in 'i3_image.c'.
    """
    n_sub = upsampling
    n_pad = padding

    cut_start = (n_pad/2)*n_sub + n_sub/2;
    cut_step = n_sub;
    
    return i3_image_dsample_cut_into(psf, stamp_size, cut_start, cut_start, cut_step)

def demo():    
    stamp_size = 32
    upsampling = 5
    padding = 0
    core_size = 7
     
    hres_size = stamp_size*upsampling
    star = np.zeros((hres_size, hres_size))
    star[hres_size/2-core_size:hres_size/2+core_size,
         hres_size/2-core_size:hres_size/2+core_size] = 1.
     
    pix_star = i3_fourier_conv_kernel(stamp_size, upsampling, padding, star)
    pix_psf = i3_downsample(pix_star, upsampling, stamp_size, padding)
    psf = i3_downsample(star, upsampling, stamp_size, padding)
     
    subplot(221)
    imshow(star)
    title('Without pixel integration before downsampling')
    subplot(222)
    imshow(pix_star)
    title('With pixel integration before downsampling')
     
    subplot(223)
    imshow(psf)
    title('Without pixel integration after downsampling')
    subplot(224)
    imshow(pix_psf)
    title('With pixel integration after downsampling')


