import argparse
import errno
import sys
import os
import glob
import numpy as np
import pdb

def main(args):

    launchers = glob.glob('%s/DECam_*' % args.launcher_folder)
    N = len(launchers)
    Ns = int(np.ceil(N / (1. * args.number_of_jobs)))

    counter = 0
    while counter < N:
        for idx in xrange(Ns):
            submission_file = 'submitter_%d.sh' % idx
            print "Writing ", submission_file
            f = open(submission_file, 'w')
            head = 'cd /direct/astro+astronfs01/workarea/mhirsch/projects/psf_analysis/launchers2\nwq sub -b'
            for i in xrange(N):
                head = head + ' ' + launchers[i].split('/')[-1]
                counter += 1

            f.write(head)
            f.close()


if __name__=="__main__":

    # Set up and parse the command line arguments using the nice Python argparse module                                                                                                                       
    description = "Create submission scripts for launching analysis jobs on BNL"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('launcher_folder', type=str, help='folder with launcher scripts')
    parser.add_argument('--number_of_jobs', type=int, default=10000, help='How many jobs per submission sciprt')

    args = parser.parse_args()

    main(args)

