# Todo:
# Sort out corrupt file handling
# Think of what to do with data next, computation of 2pt statistics and integration of Athena code
# Make cogs account and install im3shape and galsim for cogs users

import os
import sys
import pdb
import numpy as np
import galsim
import galsim.des
import fitsio
import glob
import shutil
import argparse
import math
import im3shape_psf_tools as i3

def get_radialProfile(image, x0=None, y0=None, fwxm=None):
    """
    Gives the radially averaged profile. If no center is specified, it
    is assumed that the object is centered. The optional argument fwxm
    can be specified if the radial profile only up to fwxm needs to be
    computed.
    """
    nx = image.shape[0]
    ny = image.shape[1]

    if x0 is None:
        x0 = nx/2
    if y0 is None:
        y0 = ny/2
    
    # start by determining radii of all pixels from center
    y, x = np.indices((nx, ny))
    rs = np.sqrt((x - x0)**2 + (y - y0)**2)

    ind = np.argsort(rs.flatten()) 
    sim = image.flatten()[ind]
    srs = rs.flatten()[ind]
    urs = np.unique(rs)

    if fwxm is None:    
        profile = [np.mean(sim[np.where(srs == r)]) for r in urs]
    else:
        idx = np.where(sim < fwxm * sim.max())[0][0]
        profile = [np.mean(sim[np.where(srs == r)]) for r in urs[0:idx]]      
        urs = urs[0:idx]
        
    return urs, profile

def get_FWHM(image, x0, y0, fwxm=0.5, upsampling=1, radialProfile=True):
    """
    Computes the FWHM of an image. Per default, it computes the
    radial averaged profile for determining the FWHM. Alternatively,
    it computes the FWHMs of the profiles along the x and y axes in +
    and - direction (total of 4) and returns their average as an
    estimator of the FWHM.    
    """
    x0 = np.floor(x0)
    y0 = np.floor(y0)

    profile_x = image[int(x0), :]
    profile_y = image[:, int(y0)]

    max_val = image[int(x0), int(y0)]
    cut_val = max_val * fwxm

    if radialProfile:
        radii, profile = get_radialProfile(image, x0, y0, fwxm)

        diff = abs(profile - cut_val)
         
        # fhwm code from Tomek
        f1 = 0.
        f2 = 0.
        x1 = 0
        x2 = 0
      
        x1 = np.argmin(diff)
        f1 = profile[x1]
         
        if( f1 < cut_val ):  x2 = x1+1
        else:       x2 = x1-1
        f2 = profile[x2];
         
        a = (f1-f2)/(radii[x1] - radii[x2])
        b = f1 - a*radii[x1];
        x3 = (cut_val - b)/a;
         
        fwhm = (2.* x3) / upsampling

    else:
        fwhms = []
        for i in range(4):
            if i == 0:
                profile = profile_x[int(y0)::]
                dc0 = int(x0) - x0
            if i == 1:
                profile = profile_x[0:int(y0)+1][::-1]
                dc0 = -int(x0) + x0
            if i == 2:
                profile = profile_y[int(x0)::]
                dc0 = int(y0) - y0
            if i == 3:
                profile = profile_y[0:int(x0)+1][::-1]
                dc0 = -int(y0) + y0
         
            diff = abs(profile - cut_val)
         
            # fhwm code from Tomek
            f1 = 0.
            f2 = 0.
            x1 = 0
            x2 = 0
            
            x1 = np.argmin(diff)
            f1 = profile[x1]
         
            if( f1 < cut_val ):  x2 = x1+1
            else:       x2 = x1-1
            f2 = profile[x2];
         
            a = (f1-f2)/(x1 - x2)
            b = f1 - a*x1;
            x3 = (cut_val - b)/a;
         
            fwhms.append(2.* (dc0 + x3))

        fwhm =  np.mean(np.array(fwhms))/upsampling

    return fwhm

def get_obj_params(obj, weight=None, upsampling=1.):
    """
    @brief
    Computes PSF parameters from potentially higher-resolved PSF images.
    In particular it computes e1, e2 and FWHM for each PSF in the PSF array.
    @psfs array of i3_image images of the PSF with correct resolution (should have been used earlier for fitting)
    @return psf_params - struct that contains a list [fwhm, e1, e2] for each PSF within the PSF array
    """

    # create a galsim object
    Image_obj = galsim.Image(np.ascontiguousarray(obj.astype(np.float64)))
    if weight != None:
        Image_weight = galsim.Image(np.ascontiguousarray(weight.astype(np.float64)))
        # run adaptive moments code
        try:
            result = galsim.hsm.FindAdaptiveMom(Image_obj, weight=Image_weight, strict=False)
        except:
            result = galsim.hsm.ShapeData()            
    else:
        result = galsim.hsm.FindAdaptiveMom(Image_obj, strict=False)
    # set those pixels to zero which are assigned to other objects
    try:
        obj[weight==0] = 0
        x = result.centroid.x
        y = result.centroid.y
        result.observed_fwhm = get_FWHM(obj, y, x, fwxm=0.5, upsampling=upsampling)   
    except:
        result.observed_fwhm = -1.

    return result

def getPSFExarray(psfex, x_image, y_image, wcs=None, nsidex=32, nsidey=32, upsampling=1, flux=1., offset=None):
    """Return an image of the PSFEx model of the PSF as a NumPy array.

    Arguments
    ---------
    psfex       A galsim.des.PSFEx instance opened using, for example,
                `psfex = galsim.des.DES_PSFEx(psfex_file_name)`.
    x_image     Floating point x position on image [pixels]
    y_image     Floating point y position on image [pixels]

    nsidex      Size of PSF image along x [pixels]
    nsidey      Size of PSF image along y [pixels]
    upsampling  Upsampling (see Zuntz et al 2013)
    """
    import galsim
    image = galsim.ImageD(nsidex, nsidey)
    image_pos = galsim.PositionD(float(x_image), float(y_image))
    
    # See Mike's example script within GalSim: draw_psf.py
    x = x_image
    y = y_image
    if nsidex%2 == 0:
        x += 0.5   # + 0.5 to account for even-size postage stamps
    if nsidey%2 == 0:
        y += 0.5
    ix = int(math.floor(x+0.5))  # round to nearest pixel
    iy = int(math.floor(y+0.5))
    dx = x-ix
    dy = y-iy
    offset = galsim.PositionD(float(dx),float(dy))

    psf = psfex.getPSF(image_pos)#, scale=1.)
    psf.setFlux(float(flux))
    if wcs != None:
        psf.draw(image, wcs=wcs.local(image_pos=image_pos), offset=offset)
    else:
        psf.draw(image, scale=1. / upsampling, offset=offset)
    image.setCenter(ix,iy)
    return image.array

def getShapeletPSFarray(des_shapelet, x_image, y_image, wcs=None, nsidex=32, nsidey=32, upsampling=1, flux=1., offset=None):
    """Return an image of the PSFEx model of the PSF as a NumPy array.

    Arguments
    ---------
    des_shapelet      A galsim.des.DES_Shapelet instance opened using, for example,
                      `des_shapelet = galsim.des.DES_Shapelet(shapelet_file_name)`.
                      Usually stored as *_fitpsf.fits files.
    x_image           Floating point x position on image [pixels]
    y_image           Floating point y position on image [pixels]

    nsidex            Size of PSF image along x [pixels]
    nsidey            Size of PSF image along y [pixels]
    upsampling        Upsampling (see Zuntz et al 2013)
    """
    import galsim
    image = galsim.ImageD(nsidex, nsidey)
    image_pos = galsim.PositionD(float(x_image), float(y_image))

    # See Mike's example script within GalSim: draw_psf.py
    x = x_image
    y = y_image
    if nsidex%2 == 0:
        x += 0.5   # + 0.5 to account for even-size postage stamps
    if nsidey%2 == 0:
        y += 0.5
    ix = int(math.floor(x+0.5))  # round to nearest pixel
    iy = int(math.floor(y+0.5))
    dx = x-ix
    dy = y-iy
    offset = galsim.PositionD(float(dx),float(dy))

    psf = des_shapelet.getPSF(image_pos)#, scale=1.)
    psf.setFlux(float(flux))
    if wcs != None:
        psf.draw(image, wcs=wcs.local(image_pos=image_pos), offset=offset)
    else:
        psf.draw(image, scale=1. / upsampling, offset=offset)
    image.setCenter(ix,iy)
    return image.array

def getIm3shapePSFarray(psfex, x_image, y_image, wcs=None, nsidex=32, nsidey=32, upsampling=1, flux=1., offset=None):
    """
    Tries to mimick the behaviour of im3shape's PSF handling.
    """
    import copy
    psfex_i3 = copy.deepcopy(psfex)
    if nsidex == nsidey:
        stamp_size = nsidex
    else:
        raise IOError("Size in x and y direction must be equal")
    
    psfex_i3.sample_scale = psfex_i3.sample_scale * upsampling
    psfex_image = getPSFExarray(psfex_i3, x_image, y_image, wcs=wcs, nsidex=nsidex*upsampling, nsidey=nsidey*upsampling, upsampling=1, flux=flux*(upsampling**2), offset=offset)

    padding = 0 # hard-wired for now
    i3_psf = i3.i3_fourier_conv_kernel(stamp_size, upsampling, padding, psfex_image)
    psf = i3.i3_downsample(i3_psf, upsampling, stamp_size, padding)
    
    return psf

class SextractorDataCube(object):
    def __init__(self, nstars, cat):
        self.cat = cat
        self.data = np.zeros(nstars, dtype=[('NUMBER','i4'),
                                            ('X_IMAGE','f8'),
                                            ('Y_IMAGE','f8'),
                                            ('RA','f8'),
                                            ('DEC','f8'),
                                            ('X2_IMAGE','f8'),
                                            ('Y2_IMAGE','f8'),
                                            ('XY_IMAGE','f8'),
                                            ('X2_WORLD','f8'),
                                            ('Y2_WORLD','f8'),
                                            ('XY_WORLD','f8'),
                                            ('FWHM_WORLD','f4'),
                                            ('MAG_AUTO','f4'),
                                            ('MAG_PSF','f4'),
                                            ('FLUX_PSF','f4'),
                                            ('CHI2_PSF','f4'),
                                            ('SPREAD_MODEL','f4'),
                                            ('SPREADERR_MODEL','f4'),
                                            ('CLASS_STAR','f4'),
                                            ('FWHMPSF_IMAGE','f4'),
                                            ('FWHMPSF_WORLD','f4')])

        return

    def get_entry(self, sexid, key):
        sidx = np.where(self.cat[2][:]['NUMBER'] == sexid)
        return self.cat[2][sidx][key]

    def add_row(self, idx, sexid, ra, dec):

        sidx = np.where(self.cat[2][:]['NUMBER'] == sexid)
        
        self.data[idx]['NUMBER'] = self.cat[2][sidx]['NUMBER']
        self.data[idx]['X_IMAGE'] = self.cat[2][sidx]['X_IMAGE']
        self.data[idx]['Y_IMAGE'] = self.cat[2][sidx]['Y_IMAGE']
        self.data[idx]['RA'] = ra
        self.data[idx]['DEC'] = dec
        self.data[idx]['X2_IMAGE'] = self.cat[2][sidx]['X2_IMAGE'] 
        self.data[idx]['Y2_IMAGE'] = self.cat[2][sidx]['Y2_IMAGE'] 
        self.data[idx]['XY_IMAGE'] = self.cat[2][sidx]['XY_IMAGE'] 
        self.data[idx]['X2_WORLD'] = self.cat[2][sidx]['X2_WORLD'] 
        self.data[idx]['Y2_WORLD'] = self.cat[2][sidx]['Y2_WORLD'] 
        self.data[idx]['XY_WORLD'] = self.cat[2][sidx]['XY_WORLD'] 
        self.data[idx]['FWHM_WORLD'] = self.cat[2][sidx]['FWHM_WORLD'] 
        self.data[idx]['MAG_AUTO'] = self.cat[2][sidx]['MAG_AUTO'] 
        self.data[idx]['MAG_PSF'] = self.cat[2][sidx]['MAG_PSF']
        self.data[idx]['FLUX_PSF'] = self.cat[2][sidx]['FLUX_PSF']
        self.data[idx]['CHI2_PSF'] = self.cat[2][sidx]['CHI2_PSF']
        self.data[idx]['SPREAD_MODEL'] = self.cat[2][sidx]['SPREAD_MODEL']
        self.data[idx]['SPREADERR_MODEL'] = self.cat[2][sidx]['SPREADERR_MODEL']
        self.data[idx]['CLASS_STAR'] = self.cat[2][sidx]['CLASS_STAR']
        self.data[idx]['FWHMPSF_IMAGE'] = self.cat[2][sidx]['FWHMPSF_IMAGE']
        self.data[idx]['FWHMPSF_WORLD'] = self.cat[2][sidx]['FWHMPSF_WORLD']

        return       

class HsmDataCube(object):
    def __init__(self, nstars):
        self.data = np.zeros(nstars, dtype=[('NUMBER','i4'),
                                            ('flag', 'i4'),
                                            ('observed_shape_g1','f8'),
                                            ('observed_shape_g2','f8'),
                                            ('observed_fwhm','f8'),
                                            ('corrected_e1','f8'),
                                            ('corrected_e2','f8'),
                                            ('corrected_g1','f8'),
                                            ('corrected_g2','f8'),
                                            ('corrected_shape_err','f8'),
                                            ('correction_status', 'i4'),
                                            ('moments_amp','f8'),
                                            ('moments_sigma','f8'),
                                            ('moments_centroid_x','f8'),
                                            ('moments_centroid_y','f8'),
                                            ('moments_rho4','f8'),
                                            ('moments_n_iter','i4'),
                                            ('moments_status','i4'),
                                            ('resolution_factor','f8'),
                                            ('error_message','O'),
                                            ('correction_method','O'),
                                            ('meas_type','O')])

        self.data[:]['error_message'] = 'None'
        self.data[:]['correction_method'] = 'None'
        self.data[:]['meas_type'] = 'None'
        
        return 

    def update_entry(self, idx, key, value):
        self.data[idx][key] = value

    def add_row(self, idx, sexid, hsm_data):
        self.data[idx]['NUMBER'] = sexid
        self.data[idx]['observed_shape_g1'] = hsm_data.observed_shape.g1
        self.data[idx]['observed_shape_g2'] = hsm_data.observed_shape.g2
        self.data[idx]['observed_fwhm'] = hsm_data.observed_fwhm
        self.data[idx]['corrected_e1'] = hsm_data.corrected_e1
        self.data[idx]['corrected_e2'] = hsm_data.corrected_e2
        self.data[idx]['corrected_g1'] = hsm_data.corrected_g1
        self.data[idx]['corrected_g2'] = hsm_data.corrected_g2
        self.data[idx]['corrected_shape_err'] = hsm_data.corrected_shape_err
        self.data[idx]['correction_status'] = hsm_data.correction_status
        self.data[idx]['moments_amp'] = hsm_data.moments_amp
        self.data[idx]['moments_sigma'] = hsm_data.moments_sigma
        self.data[idx]['moments_centroid_x'] = hsm_data.moments_centroid.x
        self.data[idx]['moments_centroid_y'] = hsm_data.moments_centroid.y
        self.data[idx]['moments_rho4'] = hsm_data.moments_rho4
        self.data[idx]['moments_n_iter'] = hsm_data.moments_n_iter
        self.data[idx]['moments_status'] = hsm_data.moments_status
        self.data[idx]['resolution_factor'] = hsm_data.resolution_factor
        self.data[idx]['error_message'] = hsm_data.error_message
        self.data[idx]['correction_method'] = hsm_data.correction_method
        self.data[idx]['meas_type'] = hsm_data.meas_type

        return

class Mosaic(object):
    def __init__(self, nstars, stamp_size):
        self.data = np.zeros((nstars * stamp_size, 6 * stamp_size), dtype='f8')
        self.stamp_size = stamp_size
        self.nstars = nstars
        return
    
    def add_row(self, idx, segmap, mask, star, psfex, psf_i3, shapelet):
        row = np.concatenate((segmap, mask, star, psfex, psf_i3, shapelet),1)
        self.data[0+(self.stamp_size*idx):self.stamp_size*(idx+1),0:row.shape[1]] = row
        return
        
def main(args):
    if args.all_chips:
        exposure_filenames = glob.glob('%s/*_bkg.fits.fz' % args.exposure_list)
        exposure_filenames = [exposure_filename.replace('_bkg','') for exposure_filename in exposure_filenames]
    else:
        if not args.single_exposure:
            f = open(args.exposure_list)
            exposure_filenames = f.readlines()
            f.close()
        else:
            exposure_filenames = [args.exposure_list]

    for exposure_filename in exposure_filenames:
        try:
            exposure_filename = exposure_filename.rstrip()
            # Stamp sizes of cutout and PSF model image
            stamp_size = args.stamp_size
            stamp_size2 = np.floor(stamp_size / 2)
            psf_size = stamp_size * args.upsampling # no padding!
     
            # Extract exposure id and ccd number
            exp, ccd = exposure_filename.split('/')[-1].split('.fits.fz')[0].split('_')[1:]
     
            print "Processing exposure: %s_%s" % (exp, ccd)
            print "    Loading image file: ", exposure_filename
            # Load exposure and mask
            imgfits = fitsio.FITS(exposure_filename)
            img = imgfits[1].read_image()
            hdr = fitsio.FITSHDR(imgfits[1].read_header())
     
            # Load corresponding sky background
            bg_filename = exposure_filename.replace('.fits.fz', '_bkg.fits.fz')
            print "    Loading background file: ", bg_filename
            bgfits = fitsio.FITS(bg_filename)
            bg = bgfits[1].read_image()
     
            # Load corresponding sextractor catalogue
            cat_filename = exposure_filename.replace('.fits.fz', '_cat.fits')
            print "    Loading sextractor catalogue: ", cat_filename
            sexcat = fitsio.FITS(cat_filename)
     
            # Subtract sky background
            data = img - bg
     
            # Find and load corresponding segmap 
            segmap_filename = glob.glob('/direct/astro+astronfs03/esheldon/desdata/OPS/red/*/QA/DECam_%s/DECam_%s_%s_seg.fits.fz' % (exp, exp, ccd))[0]
            print "    Loading segmap file: ", segmap_filename
            segfits = fitsio.FITS(segmap_filename)
            segmap = segfits[1].read_image()
            
            # Find and load star catalogue
            psfcat_filename = os.path.join(args.shapelet_folder, 'DECam_%s' % exp, 'DECam_%s_%s_psf.fits' % (exp,ccd))
            print "    Loading star catalogue: ", psfcat_filename
            cat = fitsio.FITS(psfcat_filename)        
            # y in psf.fits catalogue is 1sr coordinate in numpy array
            # x in psf.fits catalogue is 2nd coordinate in numpy array
     
            # Find shapelet file
            shapelet_filename = os.path.join(args.shapelet_folder, 'DECam_%s' % exp, 'DECam_%s_%s_fitpsf.fits' % (exp,ccd))
            # Find PSFEx file
            psfex_filename = exposure_filename.replace('.fits.fz', '_psfcat.psf')
     
            # Create PSFEx and shapelet instances with galsim.des module
            print "    Loading psfex file: ", psfex_filename
            des_psfex = galsim.des.DES_PSFEx(psfex_filename, exposure_filename)
            print "    Loading shapelet file: ", shapelet_filename
            des_shapelet = galsim.des.DES_Shapelet(shapelet_filename)
     
            # Create WCS instance for transforming image coordinates to wcs
            img_header = galsim.FitsHeader(imgfits[1].read_header())
            wcs = galsim.FitsWCS(header=img_header)
     
            # Create filename for result file and open file for writing
            try:
                os.makedirs(os.path.join(args.output_folder, 'DECam_%s' % exp))
            except:
                pass
            result_filename = os.path.join(args.output_folder, 'DECam_%s' % exp, 'DECam_%s_%s_psf_hsm.fits' % (exp,ccd))
            try:
                shutil.move(result_filename+'.gz', result_filename + '.gz.bak')
                print "    Create backup file as result file already existed. Backup: ", (result_filename + '.gz.bak')
            except:
                pass
            try:
                shutil.move(result_filename, result_filename + '.bak')
                print "    Create backup file as result file already existed. Backup: ", (result_filename + '.bak')
            except:
                pass
            print "    Open result file: ", result_filename
            resfits = fitsio.FITS(result_filename,'rw')
                    
            nstars = len(cat[1][:]['id'])
            print "    Number of detected stars: ", nstars
            
            # Allocate memory and instantiate data cubes
            sex_cube = SextractorDataCube(nstars,sexcat)
            star_cube = HsmDataCube(nstars)
            psfex_cube = HsmDataCube(nstars)
            psfex_i3_cube = HsmDataCube(nstars)
            shapelet_cube = HsmDataCube(nstars)
     
            # Make image for mosaic, each row contains 5 stamps: segmap, mask, image, psfex, shapelet
            mosaic = Mosaic(nstars, stamp_size)
            
            for idx in xrange(nstars):
                print "    Processing %d/%d" % (idx+1, nstars)
                sexid = cat[1][idx]['id']
                flux = sex_cube.get_entry(sexid, 'FLUX_AUTO')
                x = np.round(cat[1][idx]['x'])
                y = np.round(cat[1][idx]['y'])
     
                star_cutout = data[y-stamp_size2:y+stamp_size2,
                                   x-stamp_size2:x+stamp_size2]
                segmap_cutout = segmap[y-stamp_size2:y+stamp_size2,
                                       x-stamp_size2:x+stamp_size2]
                     
                # Set those pixels to 1 that shall be taken into account for moment measurement, all others to 0
                mask_cutout = segmap_cutout.copy()
                mask_cutout[mask_cutout==0] = 1
                mask_cutout[mask_cutout==sexid] = 1
                mask_cutout[mask_cutout!=1] = 0
                     
                if cat[1][idx]['psf_flags'] == 0:
     
                    # Convert image coordinates to wcs
                    xyimg = galsim.PositionD(float(x),float(y))
                    xywcs = wcs.toWorld(xyimg)
                    ra = xywcs.ra.rad()
                    dec = xywcs.dec.rad()
     
                    # Copy sextractor catalog entries
                    sex_cube.add_row(idx, sexid, ra, dec)
     
                    # Measure star image
                    star_params = get_obj_params(star_cutout, weight=mask_cutout, upsampling=1.)
                    star_cube.add_row(idx, sexid, star_params)
                     
                    # Build model PSF for PSFEx model
                    psf_psfex = getPSFExarray(des_psfex, x, y, wcs, psf_size, psf_size, args.upsampling, flux=flux)
                    psfex_params = get_obj_params(psf_psfex, weight=None, upsampling=args.upsampling)
                    psfex_cube.add_row(idx, sexid, psfex_params)
     
                    # Build model PSF like in im3shape 
                    psfex_i3 = getIm3shapePSFarray(des_psfex, x, y, wcs, psf_size, psf_size, 5, flux=flux)
                    psfex_i3_params = get_obj_params(psfex_i3, weight=None, upsampling=args.upsampling)
                    psfex_i3_cube.add_row(idx, sexid, psfex_i3_params)
     
                    # Shapelet
                    try:
                        psf_shapelet = getShapeletPSFarray(des_shapelet, x, y, wcs, nsidex=psf_size, nsidey=psf_size, upsampling=args.upsampling, flux=flux)
                    except:
                        if (des_shapelet.bounds.xmin > x) or (des_shapelet.bounds.xmax < x) or (des_shapelet.bounds.ymin > y) or (des_shapelet.bounds.ymax < y):                    
                            print "    Failed: -- out of bounds --"
                            shapelet_cube.update_entry(idx, 'flag', 2)
                        else:
                            print "    Failed: -- unknown error --"
                            shapelet_cube.update_entry(idx, 'flag', 3)
                        # Draw
                        psf_psfex_small = getPSFExarray(des_psfex, x, y, wcs, stamp_size, stamp_size, upsampling=1, flux=flux)
                        psf_psfex_i3_small = getIm3shapePSFarray(des_psfex, x, y, wcs, stamp_size, stamp_size, upsampling=5, flux=flux)
                        psf_shapelet_small = np.zeros((stamp_size, stamp_size))
                        mask_cutout = np.zeros((stamp_size, stamp_size))
                        mosaic.add_row(idx, segmap_cutout, mask_cutout, star_cutout, psf_psfex_small, psf_psfex_i3_small, psf_shapelet_small)                    
                        continue
                     
                    # Measure moments with GalSim's hsm code
                    shapelet_params = get_obj_params(psf_shapelet, weight=None, upsampling=args.upsampling)
                    shapelet_cube.add_row(idx, sexid, shapelet_params)                
                     
                    # Update mosaic image
                    psf_psfex_small = getPSFExarray(des_psfex, x, y, wcs, stamp_size, stamp_size, upsampling=1, flux=flux)
                    psf_psfex_i3_small = getIm3shapePSFarray(des_psfex, x, y, wcs, stamp_size, stamp_size, upsampling=5, flux=flux)
                    try:
                        psf_shapelet_small = getShapeletPSFarray(des_shapelet, x, y, wcs, nsidex=stamp_size, nsidey=stamp_size, upsampling=1, flux=flux)
                    except:
                        psf_shapelet_small = np.zeros((stamp_size, stamp_size))
                    mosaic.add_row(idx, segmap_cutout, mask_cutout, star_cutout, psf_psfex_small, psf_psfex_i3_small, psf_shapelet_small)
     
                else:
                    # Draw stamp stripe before continuing
                    psf_psfex_small = getPSFExarray(des_psfex, x, y, wcs, stamp_size, stamp_size, upsampling=1, flux=flux)
                    psf_psfex_i3_small = getIm3shapePSFarray(des_psfex, x, y, wcs, stamp_size, stamp_size, upsampling=5, flux=flux)
                    try:
                        psf_shapelet_small = getShapeletPSFarray(des_shapelet, x, y, wcs, nsidex=stamp_size, nsidey=stamp_size, upsampling=1, flux=flux)
                    except:
                        psf_shapelet_small = np.zeros((stamp_size, stamp_size))
                    mask_cutout = np.zeros((stamp_size, stamp_size))
                    mosaic.add_row(idx, segmap_cutout, mask_cutout, star_cutout, psf_psfex_small, psf_psfex_i3_small, psf_shapelet_small)
     
                    star_cube.update_entry(idx, 'flag', 1)
                    psfex_cube.update_entry(idx, 'flag', 1)
                    psfex_i3_cube.update_entry(idx, 'flag', 1)
                    shapelet_cube.update_entry(idx, 'flag', 1)
                    print "    Failed: -- rejected star --"
                    continue
     
            resfits.write_table(sex_cube.data, extname='Sextractor')
            resfits.write_table(star_cube.data, extname='Stars')
            resfits.write_table(psfex_cube.data, extname='PSFEx')
            resfits.write_table(psfex_i3_cube.data, extname='PSFEx im3shape')
            resfits.write_table(shapelet_cube.data, extname='Shapelet')
            hdr['extname'] = 'Mosaic'
            resfits.write_image(mosaic.data.astype(np.float32), header=hdr)
            resfits.close()
            os.system('gzip -f %s' % result_filename)
            print "Done. Wrote results to: ", (result_filename+'.gz')
        except:
            continue
        
if __name__=="__main__":

    # Set up and parse the command line arguments using the nice Python argparse module
    description = "Analyses the stars of a DES exposure with GalSim's hsm shape measurement code."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('exposure_list', type=str, help='list of exposure filenames that need to be processed')
    parser.add_argument('shapelet_folder', type=str, help='name of folder where to find the shapelet result files. Needed for star catalogues.')
    parser.add_argument('output_folder', type=str, help='name of folder where to save the result files')
    parser.add_argument('-u', '--upsampling', type=int, default=1, help='Upsampling factor of PSF model image creation as used e.g. in im3shape')
    parser.add_argument('-s', '--stamp_size', type=int, default=32, help='Cutout stamp size')
    parser.add_argument('--single_exposure', action='store_true', help='If you hand in the filename of an exposure you want to run on rather than a text file with filenames, use this optional argument.')
    parser.add_argument('--all_chips', action='store_true', help='Whether to process all 62 chips in one go. In this case the input is only a exposure ID without ccd id. In this case exposure_list is interpreted as folder names containing single exposures.')

    args = parser.parse_args()

    main(args)
