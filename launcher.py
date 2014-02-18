import argparse
import errno
import sys
import os
import glob

def main(args):

    submission_folder = 'launchers2'
    try:
        os.makedirs(submission_folder)
    except:
        pass
    
    f = open(args.exposure_list)
    exposure_filenames = f.readlines()
    f.close()

    N = len(exposure_filenames)

    if not args.all_chips:          
        for idx, exposure_filename in enumerate(exposure_filenames):
            try:
                exposure_filename = exposure_filename.split('\n')[0]
            #        exp, ccd = exposure_filename.split('/')[-1].split('.fits.fz')[0].split('_')[1:]
                exp, ccd = exposure_filename.split('/')[-1].split('_fitpsf.fits')[0].split('_')[1:]
                exposure_filename = glob.glob('/astro/u/astrodat/data/DES/OPS/red/*/red/DECam_%s/DECam_%s_%s.fits.fz' % (exp,exp,ccd))[0]
                print "Create submission script for exposure DECam_%s_%s - %d/%d" % (exp, ccd, idx+1, N)
            
                launch_template = open('job_specs.submission').read()
                flag = '--single_exposure'
                launch_script = launch_template.format(exp_filename=exposure_filename, exp=exp, ccd=ccd, flag=flag)
                launch_output = '%s/DECam_%s_%s' % (submission_folder, exp, ccd)
                open(launch_output,'w').write(launch_script)
     
                if args.submit:
                    os.system('wq %s' % launch_output)
            except:
                continue
    else:
        for idx, exposure_filename in enumerate(exposure_filenames):
            try:
                try:
                    exposure_filename = exposure_filename.split('\n')[0]
                except:
                    pass
                try:
                    exp, ccd = exposure_filename.split('/')[-1].split('_')[1:]
                except:
                    exp = '00' + exposure_filename.split('_')[-1]

                exposure_filename = glob.glob('/astro/u/astrodat/data/DES/OPS/red/*/red/DECam_%s' % exp)[0]
                print "Create submission script for exposure DECam_%s - %d/%d" % (exp, idx+1, N)
            
                launch_template = open('job_specs.submission').read()
                flag = '--all_chips'
                ccd = 'all'
                launch_script = launch_template.format(exp_filename=exposure_filename, exp=exp, ccd=ccd, flag=flag)
                launch_output = '%s/DECam_%s' % (submission_folder, exp)
                open(launch_output,'w').write(launch_script)
     
                if args.submit:
                    os.system('wq %s' % launch_output)
            except:
                continue
        

if __name__=="__main__":

    # Set up and parse the command line arguments using the nice Python argparse module
    description = "Launch PSF analysis jobs on BNL"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('exposure_list', type=str, help='list of exposure filenames that need to be processed')
    parser.add_argument('--submit', action='store_true', help='Whether to submit with wq')
    parser.add_argument('--all_chips', action='store_true', help='Whether to process all 62 chips in one go. In this case the input is only a exposure ID without ccd id. In this case exposure_list is interpreted as folder names containing single exposures.')

    args = parser.parse_args()

    main(args)
