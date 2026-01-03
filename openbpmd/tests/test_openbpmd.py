# Other
import numpy as np
import os
import sys
import shutil
# TODO: make openbpmd an installable python package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#import openbpmd
import simulation
import analysis

# TODO: use an actual tmpdirs for the tests

def test_get_pose_score(ref_pdb = os.path.join('files','solute.pdb'),
                        ref_trj = os.path.join('files','solute.dcd'),
                        lig_resname='MOL'):
    """Tests the get_pose_score() function and checks
    if it produces the same results as the reference
    score np array.
    """
    # Run the function
    PoseScoreArr = analysis.get_pose_score(
        ref_pdb, ref_trj, lig_resname,
    )

    # Compare its outputs with a reference
    ref_arr = os.path.join('files','reference_score_arrs.npy')
    known_PoseScoreArr = np.load(ref_arr)[0]
    assert np.all(PoseScoreArr == known_PoseScoreArr)


def test_get_contact_score(ref_pdb = os.path.join('files','solute.pdb'),
                           ref_trj = os.path.join('files','solute.dcd'),
                           lig_resname='MOL'):
    """Tests the get_contact_score() function and checks
    if it produces the same results as the reference
    score np array.
    """
    # Run the function
    ContactScoreArr = analysis.get_contact_score(
        ref_pdb, ref_trj, lig_resname,
    )

    # Compare its outputs with a reference
    ref_arr = os.path.join('files','reference_score_arrs.npy')
    known_ContactScoreArr = np.load(ref_arr)[1]

    assert np.all(ContactScoreArr == known_ContactScoreArr)


def test_minimize(structure_file=os.path.join('files','solvated.rst7'),
                  parameter_file=os.path.join('files','solvated.prm7'),
                  out_dir='test_out', min_file_name='min.pdb'):
    """Test whether we can read, minimize and write an output
    structure.
    NOTE - takes about 10-20s on 1 CPU and 1 2080Ti GPU.
    """
    # Create a tmp dir for writing the structure
    os.mkdir(out_dir)

    simulation.minimize(
        parameter_file, structure_file, out_dir, min_file_name
    )
    out_file = os.path.join(out_dir, min_file_name)
    assert os.path.isfile(out_file)
    shutil.rmtree(out_dir)


def test_equilibrate(min_eq=os.path.join('files','minimized_system.pdb'),
                     structure_file=os.path.join('files','solvated.rst7'),
                     parameter_file=os.path.join('files','solvated.prm7'),
                     out_dir='test_out', eq_file_name='eq.pdb'):
    """Test the equilibrate() function of openbpmd.
    NOTE - takes about 1-2 min on 1 CPU and 1 2080Ti GPU. 
    """
    # Create a tmp dir for writing the files
    os.mkdir(out_dir)

    simulation.equilibrate(
        min_eq, parameter_file, structure_file, out_dir, eq_file_name
    )
    # check if the output file is written
    out_file = os.path.join(out_dir, eq_file_name)

    assert os.path.isfile(out_file)
    shutil.rmtree(out_dir)


def test_produce(out_dir='test_out', rep_idx=0, lig_resname='MOL',
                 structure_file=os.path.join('files','solvated.rst7'),
                 eq_structure_file=os.path.join('files','equil_system.pdb'),
                 parameter_file=os.path.join('files','solvated.prm7'),
                 hill_height=0.3,sim_time=1):
    """Test the produce() function, check if it writes all the relevant files.
    NOTE - takes about 30 min on 1 CPU and 1 2080Ti GPU.
    """
    # Create a tmp dir for writing the structure
    os.mkdir(out_dir)
    # load structure/parameter files

    rep_dir = os.path.join(out_dir,f'rep_{rep_idx}')
    if not os.path.isdir(rep_dir):
        os.mkdir(rep_dir)

    simulation.produce(out_dir, rep_idx, lig_resname, eq_structure_file,
        parameter_file, structure_file, hill_height, sim_time
    )

    # check if the output files were written
    written_files = 'trj.dcd','COLVAR.npy','sim_log.csv'
    file_paths = [ os.path.join(rep_dir, f) for f in written_files ]
    assert os.path.isfile(file_paths[0]) and \
           os.path.isfile(file_paths[1]) and \
           os.path.isfile(file_paths[2])
    shutil.rmtree(out_dir)


def test_collect_results(in_dir='files',out_dir='test_out'):
    """Test collect_results() function.
    """
    # Create a tmp dir for writing the structure
    os.mkdir(out_dir)
    analysis.collect_results(in_dir, out_dir)
    assert os.path.isfile(os.path.join(out_dir,'results.csv'))
    shutil.rmtree(out_dir)
