import os
import sys
import pdb
import numpy as np
#import galsim
#import galsim.des
import fitsio
import glob
import shutil
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import pylab
import pickle

plotID = 6
doplot = 1

exposure_list = '/media/data/DES/SVA1/psf_analysis/shapelet-001-explist.txt'
#RESPATH = '/media/data/DES/SVA1/psf_analysis/results'
RESPATH = '/direct/astro+astronfs01/workarea/mhirsch/projects/psf_analysis/results'

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'r') as f:
        return pickle.load(f)

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def pad(data, lw=1, offset=0):

    new = np.zeros((data.shape[0]+lw,data.shape[1]+lw))
    new[offset:offset+data.shape[0], offset:offset+data.shape[1]] = data

    return new

def plot_decam(data, lw=1):

    ccds = data.keys()
    
    mosaic = zeros(((data[1].shape[0]+lw)*12, (data[1].shape[1]+lw)*7))

    rows = []
    offset = 0
    for row in xrange(12):
        if row in [0,11]:
            increment = 3
            intent = 2 * data[1].shape[1]
        if row in [1,10]:
            increment = 4
            intent = 1.5 * data[1].shape[1]
        if row in [2,9]:
            increment = 5
            intent = 1 * data[1].shape[1]
        if row in [3,4,7,8]:
            increment = 6
            intent = 0.5 * data[1].shape[1]
        if row in [5,6]:
            increment = 7
            intent = 0 * data[1].shape[1]
        
        line = concatenate([pad(data[i+1],lw) for i in xrange(offset,offset+increment)],1)
        rows.append(line)

        offset += increment
        mosaic[(row * line.shape[0]):((row+1) * line.shape[0]), intent:intent+line.shape[1]] = line
    
    return pad(mosaic, lw, offset=lw)

    

def plot_pretty_subplot(res_map, density_map, title, ax):

    pylab.imshow(rebin(res_map/density_map, [32,64]), interpolation='nearest')
    pylab.colorbar()
    pylab.xlabel('x axis (x 64 pixels)')
    pylab.ylabel('y axis (x 64 pixels)')
    pylab.title(title)

if plotID == 1:
     
    if not single_exposure:
        f = open(exposure_list)
        exposure_filenames = f.readlines()
        f.close()
    else:
        exposure_filenames = [exposure_list]
     
    for exposure_filename in exposure_filenames:
        exposure_filename = exposure_filename.rstrip()
     
        # Extract exposure id and ccd number
        exp, ccd = exposure_filename.split('/')[-1].split('.fits.fz')[0].split('_')[1:]
     
        for exp in xrange(1,63):
            try:
                reshdu = fitsio.FITS('%s/DECam_%s/DECam_%s_%s_psf_hsm.fits.gz' % (RESPATH, exp, exp, ccd))
            except:
                continue
           
elif plotID == 2:

    #exps = glob.glob('/media/data/DES/SVA1/psf_analysis/results/DECam_*')
    exps = glob.glob('/direct/astro+astronfs01/workarea/mhirsch/projects/psf_analysis/results/DECam_*')
    residual_map_g1_psfex = np.zeros((2048,4096))
    residual_map_g1_shapelet = np.zeros((2048,4096))
    residual_map_g1_psfex_shapelet = np.zeros((2048,4096))

    residual_map_g2_psfex = np.zeros((2048,4096))
    residual_map_g2_shapelet = np.zeros((2048,4096))
    residual_map_g2_psfex_shapelet = np.zeros((2048,4096))

    residual_map_sigma_psfex = np.zeros((2048,4096))
    residual_map_sigma_shapelet = np.zeros((2048,4096))
    residual_map_sigma_psfex_shapelet = np.zeros((2048,4096))

    density_map = np.ones((2048,4096))

    exps = exps[:20]    
    N = 0
    for idx, exp in enumerate(exps):
        exp = exp.split('_')[-1]
        for ccd in xrange(1,63):
            try:
                resfilename = '%s/DECam_%s/DECam_%s_%02d_psf_hsm.fits.gz' % (RESPATH, exp, exp, ccd)
                print "Loading %d/%d - %s" % ((idx*62)+ccd, len(exps)*62, resfilename)
                reshdu = fitsio.FITS(resfilename)
            except:
                print "  - Failed - "
                continue
     
            xs = reshdu[1][:]['X_IMAGE']
            ys = reshdu[1][:]['Y_IMAGE']
            ras = reshdu[1][:]['RA']
            decs = reshdu[1][:]['DEC']
            fluxs = reshdu[1][:]['FLUX_PSF']
            flags = reshdu[2][:]['flag']
     
            res_g1_psfex = reshdu[2][:]['observed_shape_g1']-reshdu[3][:]['observed_shape_g1']
            res_g1_shapelet = reshdu[2][:]['observed_shape_g1']-reshdu[4][:]['observed_shape_g1']
            res_g1_psfex_shapelet = reshdu[3][:]['observed_shape_g1']-reshdu[4][:]['observed_shape_g1']
     
            res_g2_psfex = reshdu[2][:]['observed_shape_g2']-reshdu[3][:]['observed_shape_g2']
            res_g2_shapelet = reshdu[2][:]['observed_shape_g2']-reshdu[4][:]['observed_shape_g2']
            res_g2_psfex_shapelet = reshdu[3][:]['observed_shape_g2']-reshdu[4][:]['observed_shape_g2']
            
            res_sigma_psfex = reshdu[2][:]['moments_sigma']-reshdu[3][:]['moments_sigma']
            res_sigma_shapelet = reshdu[2][:]['moments_sigma']-reshdu[4][:]['moments_sigma']
            res_sigma_psfex_shapelet = reshdu[3][:]['moments_sigma']-reshdu[4][:]['moments_sigma']
            
            for i, flag in enumerate(flags):
                if flag == 0:
                    N += 1
                    xi = max(0,min(int(np.floor(xs[i])), density_map.shape[0])-1)
                    yi = max(0,min(int(np.floor(ys[i])), density_map.shape[1])-1)

                    residual_map_g1_psfex[xi,yi] = residual_map_g1_psfex[xi,yi] + res_g1_psfex[i]
                    residual_map_g1_shapelet[xi,yi] = residual_map_g1_shapelet[xi,yi] + res_g1_shapelet[i]
                    residual_map_g1_psfex_shapelet[xi,yi] = residual_map_g1_psfex_shapelet[xi,yi] + res_g1_psfex_shapelet[i]
                    residual_map_g2_psfex[xi,yi] = residual_map_g2_psfex[xi,yi] + res_g2_psfex[i]
                    residual_map_g2_shapelet[xi,yi] = residual_map_g2_shapelet[xi,yi] + res_g2_shapelet[i]
                    residual_map_g2_psfex_shapelet[xi,yi] = residual_map_g2_psfex_shapelet[xi,yi] + res_g2_psfex_shapelet[i]
                    residual_map_sigma_psfex[xi,yi] = residual_map_sigma_psfex[xi,yi] + res_sigma_psfex[i]
                    residual_map_sigma_shapelet[xi,yi] = residual_map_sigma_shapelet[xi,yi] + res_sigma_shapelet[i]
                    residual_map_sigma_psfex_shapelet[xi,yi] = residual_map_sigma_psfex_shapelet[xi,yi] + res_sigma_psfex_shapelet[i]
                    density_map[xi,yi] = density_map[xi,yi] + 1
                else:
                    continue
            
             
    pylab.figure(figsize=(22, 10), dpi=80, facecolor='w', edgecolor='k')
    ax = pylab.subplot(331)
    plot_pretty_subplot(residual_map_g1_psfex, density_map, 'e1(Stars - PSFEx)', ax)
    ax = pylab.subplot(332)
    plot_pretty_subplot(residual_map_g2_psfex, density_map, 'e2(Stars - PSFEx)', ax)
    ax = pylab.subplot(333)
    plot_pretty_subplot(residual_map_sigma_psfex, density_map, 'size(Stars - PSFEx)', ax)

    ax = pylab.subplot(334)
    plot_pretty_subplot(residual_map_g1_shapelet, density_map, 'e1(Stars - Shapelet)', ax)
    ax = pylab.subplot(335)    
    plot_pretty_subplot(residual_map_g2_shapelet, density_map, 'e2(Stars - Shapelet)', ax)
    ax = pylab.subplot(336)    
    plot_pretty_subplot(residual_map_sigma_shapelet, density_map, 'size(Stars - Shapelet)', ax)

    ax = pylab.subplot(337)
    plot_pretty_subplot(residual_map_g1_psfex_shapelet, density_map, 'e1(PSFEx - Shapelet)', ax)
    ax = pylab.subplot(338)
    plot_pretty_subplot(residual_map_g2_psfex_shapelet, density_map, 'e2(PSFEx - Shapelet)', ax)
    ax = pylab.subplot(339)
    plot_pretty_subplot(residual_map_sigma_psfex_shapelet, density_map, 'size(PSFEx - Shapelet)', ax)
  
    pylab.savefig('ccd_residual_maps.pdf')
    pylab.close()

elif plotID == 3:

    #exps = glob.glob('/media/data/DES/SVA1/psf_analysis/results/DECam_*')
    exps = glob.glob('%s/DECam_*' % RESPATH)

    #exps = exps[:2]
    res_maps_g1_psfex = {}
    res_maps_g1_psfex_i3 = {}
    res_maps_g1_shapelet = {}
    res_maps_g1_psfex_shapelet = {}
    res_maps_g1_psfex_psfex_i3 = {}

    res_maps_g2_psfex = {}
    res_maps_g2_psfex_i3 = {}
    res_maps_g2_shapelet = {}
    res_maps_g2_psfex_shapelet = {}
    res_maps_g2_psfex_psfex_i3 = {}

    res_maps_sigma_psfex = {}
    res_maps_sigma_psfex_i3 = {}
    res_maps_sigma_shapelet = {}
    res_maps_sigma_psfex_shapelet = {}
    res_maps_sigma_psfex_psfex_i3 = {}
    for ccd in xrange(1,63):

        N = 0
        residual_map_g1_psfex = np.zeros((2048,4096))
        residual_map_g1_psfex_i3 = np.zeros((2048,4096))
        residual_map_g1_shapelet = np.zeros((2048,4096))
        residual_map_g1_psfex_shapelet = np.zeros((2048,4096))
        residual_map_g1_psfex_psfex_i3 = np.zeros((2048,4096))

        residual_map_g2_psfex = np.zeros((2048,4096))
        residual_map_g2_psfex_i3 = np.zeros((2048,4096))
        residual_map_g2_shapelet = np.zeros((2048,4096))
        residual_map_g2_psfex_shapelet = np.zeros((2048,4096))
        residual_map_g2_psfex_psfex_i3 = np.zeros((2048,4096))

        residual_map_sigma_psfex = np.zeros((2048,4096))
        residual_map_sigma_psfex_i3 = np.zeros((2048,4096))
        residual_map_sigma_shapelet = np.zeros((2048,4096))
        residual_map_sigma_psfex_shapelet = np.zeros((2048,4096))
        residual_map_sigma_psfex_psfex_i3 = np.zeros((2048,4096))

        density_map = np.ones((2048,4096))

        for idx, exp in enumerate(exps):
            exp = exp.split('_')[-1]
            try:
                resfilename = '%s/DECam_%s/DECam_%s_%02d_psf_hsm.fits.gz' % (RESPATH, exp, exp, ccd)
                print "Loading %d/%d - %s" % ((idx*62)+ccd, len(exps)*62, resfilename)
                reshdu = fitsio.FITS(resfilename)
            except:
                print "  - Failed - "
                continue
     
            xs = reshdu[1][:]['X_IMAGE']
            ys = reshdu[1][:]['Y_IMAGE']
            ras = reshdu[1][:]['RA']
            decs = reshdu[1][:]['DEC']
            fluxs = reshdu[1][:]['FLUX_PSF']
            flags = reshdu[2][:]['flag']
     
            res_g1_psfex = reshdu[2][:]['observed_shape_g1']-reshdu[3][:]['observed_shape_g1']
            res_g1_psfex_i3 = reshdu[2][:]['observed_shape_g1']-reshdu[4][:]['observed_shape_g1']
            res_g1_shapelet = reshdu[2][:]['observed_shape_g1']-reshdu[5][:]['observed_shape_g1']
            res_g1_psfex_shapelet = reshdu[3][:]['observed_shape_g1']-reshdu[5][:]['observed_shape_g1']
            res_g1_psfex_psfex_i3 = reshdu[3][:]['observed_shape_g1']-reshdu[4][:]['observed_shape_g1']
     
            res_g2_psfex = reshdu[2][:]['observed_shape_g2']-reshdu[3][:]['observed_shape_g2']
            res_g2_psfex_i3 = reshdu[2][:]['observed_shape_g2']-reshdu[4][:]['observed_shape_g2']
            res_g2_shapelet = reshdu[2][:]['observed_shape_g2']-reshdu[5][:]['observed_shape_g2']
            res_g2_psfex_shapelet = reshdu[3][:]['observed_shape_g2']-reshdu[5][:]['observed_shape_g2']
            res_g2_psfex_psfex_i3 = reshdu[3][:]['observed_shape_g2']-reshdu[4][:]['observed_shape_g2']
                        
            res_sigma_psfex = reshdu[2][:]['moments_sigma']-reshdu[3][:]['moments_sigma']
            res_sigma_psfex_i3 = reshdu[2][:]['moments_sigma']-reshdu[4][:]['moments_sigma']
            res_sigma_shapelet = reshdu[2][:]['moments_sigma']-reshdu[5][:]['moments_sigma']
            res_sigma_psfex_shapelet = reshdu[3][:]['moments_sigma']-reshdu[5][:]['moments_sigma']
            res_sigma_psfex_psfex_i3 = reshdu[3][:]['moments_sigma']-reshdu[4][:]['moments_sigma']
            
            for i, flag in enumerate(flags):
                if flag == 0:
                    N += 1
                    xi = max(0,min(int(np.floor(xs[i])), density_map.shape[0])-1)
                    yi = max(0,min(int(np.floor(ys[i])), density_map.shape[1])-1)


                    residual_map_g1_psfex[xi,yi] = residual_map_g1_psfex[xi,yi] + res_g1_psfex[i]
                    residual_map_g1_psfex_i3[xi,yi] = residual_map_g1_psfex_i3[xi,yi] + res_g1_psfex_i3[i]
                    residual_map_g1_shapelet[xi,yi] = residual_map_g1_shapelet[xi,yi] + res_g1_shapelet[i]
                    residual_map_g1_psfex_shapelet[xi,yi] = residual_map_g1_psfex_shapelet[xi,yi] + res_g1_psfex_shapelet[i]
                    residual_map_g1_psfex_psfex_i3[xi,yi] = residual_map_g1_psfex_psfex_i3[xi,yi] + res_g1_psfex_psfex_i3[i]

                    residual_map_g2_psfex[xi,yi] = residual_map_g2_psfex[xi,yi] + res_g2_psfex[i]
                    residual_map_g2_psfex_i3[xi,yi] = residual_map_g2_psfex_i3[xi,yi] + res_g2_psfex_i3[i]
                    residual_map_g2_shapelet[xi,yi] = residual_map_g2_shapelet[xi,yi] + res_g2_shapelet[i]
                    residual_map_g2_psfex_shapelet[xi,yi] = residual_map_g2_psfex_shapelet[xi,yi] + res_g2_psfex_shapelet[i]
                    residual_map_g2_psfex_psfex_i3[xi,yi] = residual_map_g2_psfex_psfex_i3[xi,yi] + res_g2_psfex_psfex_i3[i]

                    residual_map_sigma_psfex[xi,yi] = residual_map_sigma_psfex[xi,yi] + res_sigma_psfex[i]
                    residual_map_sigma_psfex_i3[xi,yi] = residual_map_sigma_psfex_i3[xi,yi] + res_sigma_psfex_i3[i]
                    residual_map_sigma_shapelet[xi,yi] = residual_map_sigma_shapelet[xi,yi] + res_sigma_shapelet[i]
                    residual_map_sigma_psfex_shapelet[xi,yi] = residual_map_sigma_psfex_shapelet[xi,yi] + res_sigma_psfex_shapelet[i]
                    residual_map_sigma_psfex_psfex_i3[xi,yi] = residual_map_sigma_psfex_psfex_i3[xi,yi] + res_sigma_psfex_psfex_i3[i]

                    density_map[xi,yi] = density_map[xi,yi] + 1
                else:
                    continue
            
        res_maps_g1_psfex[ccd] = rebin(residual_map_g1_psfex/density_map, [32,64])
        res_maps_g1_psfex_i3[ccd] = rebin(residual_map_g1_psfex_i3/density_map, [32,64])
        res_maps_g1_shapelet[ccd] = rebin(residual_map_g1_shapelet/density_map, [32,64])
        res_maps_g1_psfex_shapelet[ccd] = rebin(residual_map_g1_psfex_shapelet/density_map, [32,64])
        res_maps_g1_psfex_psfex_i3[ccd] = rebin(residual_map_g1_psfex_psfex_i3/density_map, [32,64])
                
        res_maps_g2_psfex[ccd] = rebin(residual_map_g2_psfex/density_map, [32,64])
        res_maps_g2_psfex_i3[ccd] = rebin(residual_map_g2_psfex_i3/density_map, [32,64])
        res_maps_g2_shapelet[ccd] = rebin(residual_map_g2_shapelet/density_map, [32,64])
        res_maps_g2_psfex_shapelet[ccd] = rebin(residual_map_g2_psfex_shapelet/density_map, [32,64])
        res_maps_g2_psfex_psfex_i3[ccd] = rebin(residual_map_g2_psfex_psfex_i3/density_map, [32,64])
                
        res_maps_sigma_psfex[ccd] = rebin(residual_map_sigma_psfex/density_map, [32,64])
        res_maps_sigma_psfex_i3[ccd] = rebin(residual_map_sigma_psfex_i3/density_map, [32,64])
        res_maps_sigma_shapelet[ccd] = rebin(residual_map_sigma_shapelet/density_map, [32,64])
        res_maps_sigma_psfex_shapelet[ccd] = rebin(residual_map_sigma_psfex_shapelet/density_map, [32,64])
        res_maps_sigma_psfex_psfex_i3[ccd] = rebin(residual_map_sigma_psfex_psfex_i3/density_map, [32,64])
                
        if doplot:
            pylab.figure(figsize=(22, 10), dpi=80, facecolor='w', edgecolor='k')
            ax = pylab.subplot(331)
            plot_pretty_subplot(residual_map_g1_psfex, density_map, 'e1(Stars - PSFEx)', ax)
            ax = pylab.subplot(332)
            plot_pretty_subplot(residual_map_g2_psfex, density_map, 'e2(Stars - PSFEx)', ax)
            ax = pylab.subplot(333)
            plot_pretty_subplot(residual_map_sigma_psfex, density_map, 'size(Stars - PSFEx)', ax)
         
            ax = pylab.subplot(334)
            plot_pretty_subplot(residual_map_g1_shapelet, density_map, 'e1(Stars - Shapelet)', ax)
            ax = pylab.subplot(335)    
            plot_pretty_subplot(residual_map_g2_shapelet, density_map, 'e2(Stars - Shapelet)', ax)
            ax = pylab.subplot(336)    
            plot_pretty_subplot(residual_map_sigma_shapelet, density_map, 'size(Stars - Shapelet)', ax)
         
            ax = pylab.subplot(337)
            plot_pretty_subplot(residual_map_g1_psfex_shapelet, density_map, 'e1(PSFEx - Shapelet)', ax)
            ax = pylab.subplot(338)
            plot_pretty_subplot(residual_map_g2_psfex_shapelet, density_map, 'e2(PSFEx - Shapelet)', ax)
            ax = pylab.subplot(339)
            plot_pretty_subplot(residual_map_sigma_psfex_shapelet, density_map, 'size(PSFEx - Shapelet)', ax)

            plotname = 'ccd_%02d_residual_maps.pdf' % ccd
            pylab.savefig(plotname)
            pylab.close()
            print "\n Saved figure to ", plotname

        # pickle intermediate results
        save_obj(res_maps_g1_psfex, 'res_maps_g1_psfex')
        save_obj(res_maps_g1_psfex_i3, 'res_maps_g1_psfex_i3')
        save_obj(res_maps_g1_shapelet, 'res_maps_g1_shapelet')
        save_obj(res_maps_g1_psfex_shapelet, 'res_maps_g1_psfex_shapelet')
        save_obj(res_maps_g1_psfex_psfex_i3, 'res_maps_g1_psfex_psfex_i3')
        
        save_obj(res_maps_g2_psfex, 'res_maps_g2_psfex')
        save_obj(res_maps_g2_psfex_i3, 'res_maps_g2_psfex_i3')
        save_obj(res_maps_g2_shapelet, 'res_maps_g2_shapelet')
        save_obj(res_maps_g2_psfex_shapelet, 'res_maps_g2_psfex_shapelet')
        save_obj(res_maps_g2_psfex_psfex_i3, 'res_maps_g2_psfex_psfex_i3')
        
        save_obj(res_maps_sigma_psfex, 'res_maps_sigma_psfex')
        save_obj(res_maps_sigma_psfex_i3, 'res_maps_sigma_psfex_i3')
        save_obj(res_maps_sigma_shapelet, 'res_maps_sigma_shapelet')
        save_obj(res_maps_sigma_psfex_shapelet, 'res_maps_sigma_psfex_shapelet')
        save_obj(res_maps_sigma_psfex_psfex_i3, 'res_maps_sigma_psfex_psfex_i3')



elif plotID == 4:

    #exps = glob.glob('/media/data/DES/SVA1/psf_analysis/results/DECam_*')
    exps = glob.glob('%s/DECam_*' % RESPATH)
    resfile = 'global_stats'

    Ns = []
    filters = []
    ras = []
    decs = []
    for idx, exp in enumerate(exps):
        print "Processing %d/%d" % (idx, len(exps))
        for ccd in xrange(1,63):
            exp = exp.split('_')[-1]
            try:
                resfilename = '%s/DECam_%s/DECam_%s_%02d_psf_hsm.fits.gz' % (RESPATH, exp, exp, ccd)
                reshdu = fitsio.FITS(resfilename)
            except:
                print "  - Failed when processing: %s " % resfilename
                continue
     
            xs = reshdu[1][:]['X_IMAGE']
            ys = reshdu[1][:]['Y_IMAGE']
            ra = reshdu[1][:]['RA']
            dec = reshdu[1][:]['DEC']
            class_star = reshdu[1][:]['CLASS_STAR']
            
            Ns.append(np.sum(class_star > 0.9))            
            filters.append(reshdu[6].read_header()['FILTER'])
            ras.append(np.median(ra))
            decs.append(np.median(dec))

        if  idx % 10 == 0:
            dict = {}
            dict['Ns'] = Ns
            dict['ras'] = ras
            dict['decs'] = decs
            dict['filters'] = filters
            print "  - Saving results to ", resfile
            save_obj(dict, resfile)

    dict = {}
    dict['Ns'] = Ns
    dict['ras'] = ras
    dict['decs'] = decs
    dict['filters'] = filters
    
    save_obj(dict, resfile)
            
elif plotID == 5:

    #exps = glob.glob('/media/data/DES/SVA1/psf_analysis/results/DECam_*')
    exps = glob.glob('%s/DECam_*' % RESPATH)
    band = 'i'
    resfile = 'full_catalogue_%s' % band

    xs = []
    ys = []
    ra = []
    dec = []
    class_star = []

    e1_star = []
    e2_star = []
    sigma_star = []
    flag_star = []
    
    e1_psfex = []
    e2_psfex = []
    sigma_psfex = []
    flag_psfex = []
    
    e1_psfex_i3 = []
    e2_psfex_i3 = []
    sigma_psfex_i3 = []
    flag_psfex_i3 = []
    
    e1_shapelet = []
    e2_shapelet = []
    sigma_shapelet = []
    flag_shapelet = []

    counter = 0
    for idx, exp in enumerate(exps):
        print "Processing %d/%d" % (idx, len(exps))
        for ccd in xrange(1,63):
            exp = exp.split('_')[-1]
            try:
                resfilename = '%s/DECam_%s/DECam_%s_%02d_psf_hsm.fits.gz' % (RESPATH, exp, exp, ccd)
                reshdu = fitsio.FITS(resfilename)
            except:
                print "  - Failed when processing: %s " % resfilename
                continue

            if reshdu[6].read_header()['FILTER'][0] == band:
                if ccd == 1:
                    counter += 1
                
                xs.append(reshdu[1][:]['X_IMAGE'])
                ys.append(reshdu[1][:]['Y_IMAGE'])
                ra.append(reshdu[1][:]['RA'])
                dec.append(reshdu[1][:]['DEC'])
                class_star.append(reshdu[1][:]['CLASS_STAR'])
            
                e1_star.append(reshdu[2][:]['observed_shape_g1'])
                e2_star.append(reshdu[2][:]['observed_shape_g2'])
                sigma_star.append(reshdu[2][:]['moments_sigma'])
                flag_star.append(reshdu[2][:]['flag'])

                e1_psfex.append(reshdu[3][:]['observed_shape_g1'])
                e2_psfex.append(reshdu[3][:]['observed_shape_g2'])
                sigma_psfex.append(reshdu[3][:]['moments_sigma'])
                flag_psfex.append(reshdu[3][:]['flag'])
            
                e1_psfex_i3.append(reshdu[4][:]['observed_shape_g1'])
                e2_psfex_i3.append(reshdu[4][:]['observed_shape_g2'])
                sigma_psfex_i3.append(reshdu[4][:]['moments_sigma'])
                flag_psfex_i3.append(reshdu[4][:]['flag'])
            
                e1_shapelet.append(reshdu[5][:]['observed_shape_g1'])
                e2_shapelet.append(reshdu[5][:]['observed_shape_g2'])
                sigma_shapelet.append(reshdu[5][:]['moments_sigma'])
                flag_shapelet.append(reshdu[5][:]['flag'])                
                
            else:
                print "  - Skip"
                break

        if  counter % 100 == 0:
            dict = {}

            dict['xs'] = xs
            dict['ys'] = ys
            dict['ra'] = ra
            dict['dec'] = dec
            dict['class_star'] = class_star
                
            dict['e1_star'] = e1_star
            dict['e2_star'] = e2_star
            dict['sigma_star'] = sigma_star
            dict['flag_star'] = flag_star
                
            dict['e1_psfex'] = e1_psfex
            dict['e2_psfex'] = e2_psfex
            dict['sigma_psfex'] = sigma_psfex
            dict['flag_psfex'] = flag_psfex

            dict['e1_psfex_i3'] = e1_psfex_i3
            dict['e2_psfex_i3'] = e2_psfex_i3
            dict['sigma_psfex_i3'] = sigma_psfex_i3
            dict['flag_psfex_i3'] = flag_psfex_i3
                
            dict['e1_shapelet'] = e1_shapelet
            dict['e2_shapelet'] = e2_shapelet
            dict['sigma_shapelet'] = sigma_shapelet
            dict['flag_shapelet'] = flag_shapelet

            filename = resfile + '_%d' % counter
            print "  - Saving results to ", resfile
            save_obj(dict, resfile)

            counter += 1


    dict = {}
    dict['xs'] = xs
    dict['ys'] = ys
    dict['ra'] = ra
    dict['dec'] = dec
    dict['class_star'] = class_star
    
    dict['e1_star'] = e1_star
    dict['e2_star'] = e2_star
    dict['sigma_star'] = sigma_star
    dict['flag_star'] = flag_star

    dict['e1_psfex'] = e1_psfex
    dict['e2_psfex'] = e2_psfex
    dict['sigma_psfex'] = sigma_psfex
    dict['flag_psfex'] = flag_psfex
    
    dict['e1_psfex_i3'] = e1_psfex_i3
    dict['e2_psfex_i3'] = e2_psfex_i3
    dict['sigma_psfex_i3'] = sigma_psfex_i3
    dict['flag_psfex_i3'] = flag_psfex_i3

    dict['e1_shapelet'] = e1_shapelet
    dict['e2_shapelet'] = e2_shapelet
    dict['sigma_shapelet'] = sigma_shapelet
    dict['flag_shapelet'] = flag_shapelet

    save_obj(dict, resfile)
            
elif plotID == 6:

    #exps = glob.glob('/media/data/DES/SVA1/psf_analysis/results/DECam_*')
    exps = glob.glob('%s/DECam_*' % RESPATH)
    band = 'i'
    resfile = 'full_catalogue_%s.txt' % band
    
    f = open(resfile,'wb')
    headerline = '#{exp}_{ccd} NUMBER X_IMAGE Y_IMAGE RA DEC MAG_AUTO MAG_PSF FLUX_PSF CLASS_STAR stars_observed_shape_g1 stars_observed_shape_g2 stars_moments_sigma stars_moments_amp stars_flag psfex_observed_shape_g1 psfex_observed_shape_g2 psfex_moments_sigma psfex_moments_amp psfex_flag psfex_i3_observed_shape_g1 psfex_i3_observed_shape_g2 psfex_i3_moments_sigma psfex_i3_moments_amp psfex_i3_flag shapelet_observed_shape_g1 shapelet_observed_shape_g2 shapelet_moments_sigma shapelet_moments_amp shapelet_flag '
    f.write(headerline)

    for idx, exp in enumerate(exps):
        print "Processing %d/%d" % (idx, len(exps))
        for ccd in xrange(1,63):
            exp = exp.split('_')[-1]
            try:
                resfilename = '%s/DECam_%s/DECam_%s_%02d_psf_hsm.fits.gz' % (RESPATH, exp, exp, ccd)
                reshdu = fitsio.FITS(resfilename)
                
            except:
                print "  - Failed when processing: %s " % resfilename
                continue

            if reshdu[6].read_header()['FILTER'][0] == band:

                data = np.vstack((np.repeat(np.int(exp), reshdu[1][:]['NUMBER'].shape),
                                  np.repeat(np.int(ccd), reshdu[1][:]['NUMBER'].shape),
                                  reshdu[1][:]['NUMBER'],
                                  reshdu[1][:]['X_IMAGE'],
                                  reshdu[1][:]['Y_IMAGE'],
                                  reshdu[1][:]['RA'],
                                  reshdu[1][:]['DEC'],
                                  reshdu[1][:]['MAG_AUTO'],
                                  reshdu[1][:]['MAG_PSF'],
                                  reshdu[1][:]['FLUX_PSF'],
                                  reshdu[1][:]['CLASS_STAR'],
                                  reshdu[2][:]['observed_shape_g1'],
                                  reshdu[2][:]['observed_shape_g2'],
                                  reshdu[2][:]['moments_sigma'],
                                  reshdu[2][:]['moments_amp'],
                                  reshdu[2][:]['flag'],
                                  reshdu[3][:]['observed_shape_g1'],
                                  reshdu[3][:]['observed_shape_g2'],
                                  reshdu[3][:]['moments_sigma'],
                                  reshdu[3][:]['moments_amp'],
                                  reshdu[3][:]['flag'],            
                                  reshdu[4][:]['observed_shape_g1'],
                                  reshdu[4][:]['observed_shape_g2'],
                                  reshdu[4][:]['moments_sigma'],
                                  reshdu[4][:]['moments_amp'],
                                  reshdu[4][:]['flag'],            
                                  reshdu[5][:]['observed_shape_g1'],
                                  reshdu[5][:]['observed_shape_g2'],
                                  reshdu[5][:]['moments_sigma'],
                                  reshdu[5][:]['moments_amp'],
                                  reshdu[5][:]['flag']))
                                 
                np.savetxt(f, data.T, fmt='%d %d %d %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %d %.8e %.8e %.8e %.8e %d %.8e %.8e %.8e %.8e %d %.8e %.8e %.8e %.8e %d')
                                           
            else:
                print "  - Skip"
                break

    f.close()


exp
ccd
NUMBER
X_IMAGE
Y_IMAGE
RA
DEC
MAG_AUTO
MAG_PSF
FLUX_PSF
CLASS_STAR
observed_shape_g1
observed_shape_g2
moments_sigma
moments_amp
flag
observed_shape_g1
observed_shape_g2
moments_sigma
moments_amp
flag
observed_shape_g1
observed_shape_g2
moments_sigma
moments_amp
flag
observed_shape_g1
observed_shape_g2
moments_sigma
moments_amp
flag
