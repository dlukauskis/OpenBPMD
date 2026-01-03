#!/usr/bin/env python
descriptn = \
    """
    OpenBPMD - an open source implementation of Binding Pose Metadynamics
    (BPMD) with OpenMM. Replicates the protocol as described by
    Clark et al. 2016 (DOI: 10.1021/acs.jctc.6b00201).

    Runs ten 10 ns metadynamics simulations that biases the RMSD of the ligand.

    The stability of the ligand is calculated using the ligand RMSD (PoseScore)
    and the persistence of the original noncovalent interactions between the
    protein and the ligand (ContactScore). Stable poses have a low RMSD and
    a high fraction of the native contacts preserved until the end of the
    simulation.

    A composite score is calculated using the following formula:
    CompScore = PoseScore - 5 * ContactScore
    """

# Own functions for analysis and production
import openbpmd.analysis
import openbpmd.simulation

# The rest
import argparse
import numpy as np
import mdtraj as md
import pandas as pd
import os


__author__ = 'Dominykas Lukauskis'
__version__ = '1.0.1'
__email__ = 'lukauskisdominykas@gmail.com'


def main(args):
    """Main entry point of the app. Takes in argparse.Namespace object as
    a function argument. Carries out a sequence of steps required to obtain a
    stability score for a given ligand pose in the provided structure file.

    1. Load the structure and parameter files.
    2. If absent, create an output folder.
    3. Minimization up to ener. tolerance of 10 kJ/mol.
    4. 500 ps equilibration in NVT ensemble with position
       restraints on solute heavy atoms with the force 
       constant of 5 kcal/mol/A^2
    5. Run NREPs (default=10) of binding pose metadynamics simulations,
       writing trajectory files and a time-resolved BPM scores for each
       repeat.
    6. Collect results from the OpenBPMD simulations and
       write a final score for a given protein-ligand
       structure.

    Parameters
    ----------
    args.structure : str, default='solvated.rst7'
        Name of the structure file, either Amber or Gromacs format.
    args.parameters : str, default='solvated.prm7'
        Name of the parameter or topology file, either Amber or Gromacs
        format.
    args.output : str, default='.'
        Path to and the name of the output directory.
    args.lig_resname : str, default='MOL'
        Residue name of the ligand in the structure/parameter file.
    args.nreps : int, default=10
        Number of repeat OpenBPMD simulations to run in series.
    args.hill_height : float, default=0.3
        Size of the metadynamical hill, in kcal/mol.
    """
    if not os.path.isdir(f'{args.output}'):
        os.mkdir(f'{args.output}')

    # Minimize
    min_file_name = 'minimized_system.pdb'
    if not os.path.isfile(os.path.join(args.output,min_file_name)):
        print("Minimizing...")
        openbpmd.simulation.minimize(
            args.parameters, args.structure, args.output, min_file_name
        )
    min_pdb = os.path.join(args.output, min_file_name)

    # Equilibrate
    eq_file_name = 'equil_system.pdb'
    if not os.path.isfile(os.path.join(args.output, eq_file_name)):
        print("Equilibrating...")
        openbpmd.simulation.equilibrate(
            min_pdb, args.parameters, args.structure, args.output, eq_file_name
        )
    eq_pdb = os.path.join(args.output,eq_file_name)
    cent_eq_pdb = os.path.join(args.output,'centred_'+eq_file_name)
    if os.path.isfile(eq_pdb) and not os.path.isfile(cent_eq_pdb):
        # mdtraj can't use GMX TOP, so we have to specify the GRO file instead
        if args.structure.endswith('.gro'):
            mdtraj_top = args.structure
        else:
            mdtraj_top = args.parameters
        mdu = md.load(eq_pdb, top=mdtraj_top)
        mdu.image_molecules()
        mdu.save_pdb(cent_eq_pdb)

    # Run NREPS number of production simulations
    for idx in range(0, args.nreps):
        rep_dir = os.path.join(args.output,f'rep_{idx}')
        if not os.path.isdir(rep_dir):
            os.mkdir(rep_dir)

        if os.path.isfile(os.path.join(rep_dir,'bpmd_results.csv')):
            continue
        
        openbpmd.simulation.produce(
            args.output, idx, args.lig_resname, eq_pdb, args.parameters,
            args.structure, args.hill_height, 10
        )
                
        trj_name = os.path.join(rep_dir,'trj.dcd')
                
        PoseScoreArr = openbpmd.analysis.get_pose_score(
            cent_eq_pdb, trj_name, args.lig_resname
        )

        ContactScoreArr = openbpmd.analysis.get_contact_score(
            cent_eq_pdb, trj_name, args.lig_resname
        )

        # Calculate the CompScore at every frame
        CompScoreArr = np.zeros(99)
        for index in range(ContactScoreArr.shape[0]):
            ContactScore, PoseScore = ContactScoreArr[index], PoseScoreArr[index]
            CompScore = PoseScore - 5 * ContactScore
            CompScoreArr[index] = CompScore

        Scores = np.stack((CompScoreArr, PoseScoreArr, ContactScoreArr), axis=-1)

        # Save a DataFrame to CSV
        df = pd.DataFrame(Scores, columns=['CompScore', 'PoseScore',
                                           'ContactScore'])
        df.to_csv(os.path.join(rep_dir,'bpmd_results.csv'), index=False)
                
    openbpmd.analysis.collect_results(args.output, args.output)

    return None
    
    
if __name__ == "__main__":
    """ This is executed when run from the command line """
    # Parse the CLI arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=descriptn)

    parser.add_argument("-s", "--structure", type=str, default='solvated.rst7',
                        help='input structure file name (default: %(default)s)')
    parser.add_argument("-p", "--parameters", type=str, default='solvated.prm7',
                        help='input topology file name (default: %(default)s)')
    parser.add_argument("-o", "--output", type=str, default='.',
                        help='output location (default: %(default)s)')
    parser.add_argument("-lig_resname", type=str, default='MOL',
                        help='the name of the ligand (default: %(default)s)')
    parser.add_argument("-nreps", type=int, default=10,
                        help="number of OpenBPMD repeats (default: %(default)i)")
    parser.add_argument("-hill_height", type=float, default=0.3,
                        help="the hill height in kcal/mol (default: %(default)f)")

    args = parser.parse_args()
    main(args)
