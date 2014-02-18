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

try:
    athena_dirname = '/direct/astro+u/mhirsch/usr/athena/athena_1.6' #os.environ['ATHENA_DIR']
    ATHENA_BIN = os.path.join(athena_dirname,'bin','athena')
    assert os.path.exists(ATHENA_BIN), ("Could not find " + ATHENA_BIN)
    del athena_dirname
except Exception as error:
    sys.stderr.write("%s\n"%str(error))
    sys.stderr.write("You need to set ATHENA_DIR to run this script\n")
    sys.exit(1)

try:
    corr2_dirname = '/direct/astro+u/mhirsch/usr/mjarvis-read-only'
    CORR2_BIN = os.path.join(corr2_dirname,'corr2')
    assert os.path.exists(CORR2_BIN), ("Could not find " + CORR2_BIN)
    del corr2_dirname
except Exception as error:
    sys.stderr.write("%s\n"%str(error))
    sys.stderr.write("You need to set CORR2_DIR to run this script\n")
    sys.exit(1)

RESULTPATH = "/global/project/projectdirs/des/wl/desdata/wlpipe/hsm-psfex-011-1"
CODEPATH = "/direct/astro+astronfs01/workarea/mhirsch/projects/psf_analysis"
CORR2_CONFIG_RHO1_TEMPLATE = "%s/corr2.config.rho1.template" % CODEPATH
CONFIG_RHO1_TEMPLATE = "%s/config.rho1.template" % CODEPATH
CONFIG_RHO2_TEMPLATE = "%s/config.rho2.template" % CODEPATH
THRESHOLD_DELTAE = 0.1

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def compute_rhostats(rho1_filename, rho2_filename, corr2=False):

    output_rho1_filename = rho1_filename.replace('rho1.cat', 'rho1.xi')
    output_rho2_filename = rho2_filename.replace('rho2.cat', 'rho2.xi')

    # Create config files
    config_template = open(CONFIG_RHO1_TEMPLATE).read()    
    config_script = config_template.format(cat1=rho1_filename)
    config_rho1_filename = rho1_filename + '.conf'
    open(config_rho1_filename,'w').write(config_script)
     
    config_template = open(CONFIG_RHO2_TEMPLATE).read()    
    config_script = config_template.format(cat1=rho1_filename,
                                           cat2=rho2_filename)
    config_rho2_filename = rho2_filename + '.conf'
    open(config_rho2_filename,'w').write(config_script)

    # Generate catalog specific config file
    os.system("%s -q -c %s --out_xi %s" % (ATHENA_BIN, config_rho1_filename, output_rho1_filename))
    os.system("%s -q -c %s --out_xi %s" % (ATHENA_BIN, config_rho2_filename, output_rho2_filename))

    if corr2:
        output_rho1_filename = rho1_filename.replace('rho1.cat', 'rho1.xi.corr2')

        # Create config files
        config_template = open(CORR2_CONFIG_RHO1_TEMPLATE).read()    
        config_script = config_template.format(cat1=rho1_filename,out1=output_rho1_filename)
        config_rho1_filename = rho1_filename + '.conf.corr2'
        open(config_rho1_filename,'w').write(config_script)

        # Generate catalog specific config file
        os.system("%s %s" % (CORR2_BIN, config_rho1_filename))
        

def main(args):
    if args.all_chips:
        exposure_filenames = glob.glob('%s/%s/*_psf_hsm.fits.gz' % (args.result_folder, args.exposure_list))
    else:
        if not args.single_exposure:
            f = open(args.exposure_list)
            exposure_filenames = f.readlines()
            f.close()
        else:
            exposure_filenames = [args.exposure_list]

    for exposure_filename in exposure_filenames:
        try:
            result_filename = exposure_filename.rstrip()

            # Extract exposure id and ccd number
            exp, ccd = exposure_filename.split('/')[-1].split('_psf_hsm.fits.gz')[0].split('_')[1:]

            # Load result catalogue
            #print "Processing exposure: %s_%s" % (exp, ccd)
            #print "    Loading image file: ", exposure_filename
            hdu = fitsio.FITS(result_filename)
            band = hdu[6].read_header()['FILTER'][0]

            # Write necessary catalogues for computing rho stats with
            # Athena, incl. band info in filename for easier post-processing
            try:
                os.makedirs(os.path.join(args.output_folder, 'DECam_%s' % exp))
            except:
                pass
            
            filestem = os.path.join(args.output_folder, 'DECam_%s' % exp, 'DECam_%s_%s' % (exp,ccd))
            rho1_filename = '%s.%s.%s.rho1.cat' % (filestem, band, args.psf_model)
            rho2_filename = '%s.%s.%s.rho2.cat' % (filestem, band, args.psf_model)
            if args.psf_model == 'psfex':
                idx = 3
            elif args.psf_model == 'psfex_im3shape':
                idx = 4
            elif args.psf_model == 'shapelet':
                idx = 5
            else:
                raise IOError('Unknown PSF model', args.psf_method)

            ra = hdu[1][:]['RA']
            dec = hdu[1][:]['DEC']
            e1 = hdu[idx][:]['observed_shape_g1']
            e2 = hdu[idx][:]['observed_shape_g2']
            de1 = hdu[2][:]['observed_shape_g1'] - hdu[idx][:]['observed_shape_g1']
            de2 = hdu[2][:]['observed_shape_g2'] - hdu[idx][:]['observed_shape_g2']

            # Include only those objects that haven't successfully
            # passed hsm's shape measurement and also reject outliers
            selection = (  (hdu[2][:]['flag'] == 0)
                         * (hdu[idx][:]['flag'] == 0)
                         * (abs(de1) < THRESHOLD_DELTAE)
                         * (abs(de2) < THRESHOLD_DELTAE))

            ra = ra.compress(selection==1) * 180/np.pi
            dec = dec.compress(selection==1) * 180/np.pi
            e1 = e1.compress(selection==1)
            e2 = e2.compress(selection==1)
            de1 = de1.compress(selection==1)
            de2 = de2.compress(selection==1)

            #print "    Write rho1 catalogue to: ", rho1_filename
            np.savetxt(rho1_filename, np.vstack((ra, dec, de1, de2, np.repeat(1.,len(ra)))).T, fmt="%.8f\t%.8f\t%.8f\t%.8f\t%.8f")
            #print "    Write rho2 catalogue to: ", rho2_filename
            np.savetxt(rho2_filename, np.vstack((ra, dec, e1, e2, np.repeat(1.,len(ra)))).T, fmt="%.8f\t%.8f\t%.8f\t%.8f\t%.8f")

            # Call athena
            #print "    Calling athena"
            compute_rhostats(rho1_filename, rho2_filename, corr2=True)
            #print "    Done"

        except:
            continue



if __name__=="__main__":

    # Set up and parse the command line arguments using the nice Python argparse module
    description = "Analyses the stars of a DES exposure with GalSim's hsm shape measurement code."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('exposure_list', type=str, help='list of exposure filenames that need to be processed')
    parser.add_argument('result_folder', type=str, help='name of folder where the hsm  result files are')
    parser.add_argument('output_folder', type=str, help='name of folder where to save the result files')
    parser.add_argument('psf_model', type=str, help='which psf model to analye. Could be one of ''psfex'',''psfex_im3shape'', ''shapelet''')
    parser.add_argument('--single_exposure', action='store_true', help='If you hand in the filename of an exposure you want to run on rather than a text file with filenames, use this optional argument.')
    parser.add_argument('--all_chips', action='store_true', help='Whether to process all 62 chips in one go. In this case the input is only a exposure ID without ccd id. In this case exposure_list is interpreted as folder names containing single exposures.')

    args = parser.parse_args()

    main(args)
