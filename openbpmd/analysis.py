import os
import glob
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
import numpy as np
import pandas as pd

def collect_results(in_dir, out_dir):
    """A function that collects the time-resolved BPM results,
    takes the scores from last 2 ns of the simulation, averages them
    and writes that average as the final score for a given pose.

    Writes a 'results.csv' file in 'out_dir' directory.

    Parameters
    ----------
    in_dir : str
        Directory with 'rep_*' directories.
    out_dir : str
        Directory where the 'results.csv' file will be written
    """
    compList = []
    contactList = []
    poseList = []
    # find how many repeats have been run
    glob_str = os.path.join(in_dir,'rep_*')
    nreps = len(glob.glob(glob_str))
    for idx in range(0, nreps):
        f = os.path.join(in_dir,f'rep_{idx}','bpmd_results.csv')
        df = pd.read_csv(f)
        # Since we only want last 2 ns, get the index of
        # the last 20% of the data points
        last_2ns_idx = round(len(df['CompScore'].values)/5)  # round up
        compList.append(df['CompScore'].values[-last_2ns_idx:])
        contactList.append(df['ContactScore'].values[-last_2ns_idx:])
        poseList.append(df['PoseScore'].values[-last_2ns_idx:])

    # Get the means of the last 2 ns
    meanCompScore = np.mean(compList)
    meanPoseScore = np.mean(poseList)
    meanContact = np.mean(contactList)
    # Get the standard deviation of the final 2 ns
    meanCompScore_std = np.std(compList)
    meanPoseScore_std = np.std(poseList)
    meanContact_std = np.std(contactList)
    # Format it the Pandas way
    d = {'CompScore': [meanCompScore], 'CompScoreSD': [meanCompScore_std],
         'PoseScore': [meanPoseScore], 'PoseScoreSD': [meanPoseScore_std],
         'ContactScore': [meanContact], 'ContactScoreSD': [meanContact_std]}

    results_df = pd.DataFrame(data=d)
    results_df = results_df.round(3)
    results_df.to_csv(os.path.join(out_dir,'results.csv'), index=False)


def single_rep_plot(bpm_results_f):
    """
    A function that plots the OpenBPMD scores from one repeat.
    
    Parameters
    ----------
    bpm_results_f : str
        Path to the 'bpm_results.csv' file rom one OpenBPMD simulation.
    """
    pose_name, rep_id = bpm_results_f.split('/')[-3:-1]
    df = pd.read_csv(bpm_results_f)
    time_sequence = np.linspace(0,10,df.shape[0])
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                        figsize = (13, 4))
    fig.suptitle(f'Results for {pose_name}, {rep_id}')
    
    ax1.set_title('CompScore')
    ax1.plot(time_sequence, df['CompScore'], color='blue')
    ax1.set_xlabel('time(ns)')
    ax1.set_ylabel('CompScore')
    ax1.set_ylim(-6,6)

    ax2.set_title('PoseScore')
    ax2.plot(time_sequence, df['PoseScore'],color='darkorange')
    ax2.set_xlabel('time(ns)')
    ax2.set_ylabel('PoseScore')
    ax2.set_ylim(0,5)

    ax3.set_title('ContactScore')
    ax3.plot(time_sequence, df['ContactScore'],color='green')
    ax3.set_xlabel('time(ns)')
    ax3.set_ylabel('ContactScore')
    ax3.set_ylim(-0.1,1.1)
    
    plt.tight_layout()
    plt.show()

    return None


def get_contact_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the ContactScore from an OpenBPMD trajectory.

    Parameters
    ----------
    structure_file : str
        The name of the centred equilibrated system PDB file that
        was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    contact_scores : np.array
        ContactScore for every frame of the trajectory.
    """
    u = mda.Universe(structure_file, trajectory_file)

    sel_donor = f"resname {lig_resname} and not name *H*"
    sel_acceptor = f"protein and not name H* and \
                     around 5 resname {lig_resname}"

    # reference groups (first frame of the trajectory, but you could also use
    # a separate PDB, eg crystal structure)
    a_donors = u.select_atoms(sel_donor)
    a_acceptors = u.select_atoms(sel_acceptor)

    cont_analysis = contacts.Contacts(u, select=(sel_donor, sel_acceptor),
                                      refgroup=(a_donors, a_acceptors),
                                      radius=3.5)

    cont_analysis.run()
    # print number of average contacts in the first ns
    # NOTE - hard coded number of frames (100 per traj)
    frame_idx_first_ns = int(len(cont_analysis.timeseries)/10)
    first_ns_mean = np.mean(cont_analysis.timeseries[1:frame_idx_first_ns, 1])
    if first_ns_mean == 0:
        normed_contacts = cont_analysis.timeseries[1:, 1]
    else:
        normed_contacts = cont_analysis.timeseries[1:, 1]/first_ns_mean
    contact_scores = np.where(normed_contacts > 1, 1, normed_contacts)

    return contact_scores


def get_pose_score(structure_file, trajectory_file, lig_resname):
    """A function the gets the PoseScore (ligand RMSD) from an OpenBPMD
    trajectory.

    Parameters
    ----------
    'structure_file : str
        The name of the centred equilibrated system
        PDB file that was used to start the OpenBPMD simulation.
    trajectory_file : str
        The name of the OpenBPMD trajectory file.
    lig_resname : str
        Residue name of the ligand that was biased.

    Returns
    -------
    pose_scores : np.array
        PoseScore for every frame of the trajectory.
    """
    # Load an MDA universe with the trajectory
    u = mda.Universe(structure_file, trajectory_file)
    # Align each frame using the backbone as a reference
    # Calculate the RMSD of ligand heavy atoms
    r = rms.RMSD(u, select='backbone',
                 groupselections=[f'resname {lig_resname} and not name H*'],
                 ref_frame=0).run()
    # Get the PoseScores as np.array
    pose_scores = r.rmsd[1:, -1]

    return pose_scores

    
def plot_all_reps(run_dir, save_fig=False):
    """
    A function that plots the OpenBPMD scores from all repeats it can find in a 
    directory provided.
    Parameters
    ----------
    run_dir : str
        Path to the directory with 'rep_*' from a OpenBPMD run.
    save_fig : str
        If you want to save the plot, provide a path and/or name for the file.
    """
    
    pose_name = run_dir.split('/')[-1]
    rep_dirs = sorted(glob.glob(os.path.join(run_dir,'rep_*')))
    if not rep_dirs:
        raise Exception(f"No rep_* found in '{run_dir}'")
    
    # We'll store the results from 10 repeats in a 10x99 matrix.
    n_reps = len(rep_dirs)
    CompScores = np.zeros((n_reps,99))
    PoseScores = np.zeros((n_reps,99))
    ContactScores = np.zeros((n_reps,99))

    # Fill those matrices with the scores from each repeat
    for idx, rep_dir in enumerate(rep_dirs):
        f = os.path.join(rep_dir, 'bpmd_results.csv')
        df = pd.read_csv(f)
        CompScores[idx] = df['CompScore']
        PoseScores[idx] = df['PoseScore']
        ContactScores[idx] = df['ContactScore']

    # Average out the scores from all of the repeats,
    # giving a mean of the scores at each frame of the trajectory
    averagedCompScore = np.array([ np.mean(CompScores[:,i]) for i in range(0,99) ])
    averagedPoseScore = [ np.mean(PoseScores[:,i]) for i in range(0,99) ]
    averagedContactScore = [ np.mean(ContactScores[:,i]) for i in range(0,99) ]
    # Get the standard deviation for the CompScore
    CompScore_stddev = np.array([ np.std(CompScores[:,i]) for i in range(0,99) ])
    # Get the standard deviation for the PoseScore
    PoseScore_stddev = np.array([ np.std(PoseScores[:,i]) for i in range(0,99) ])
    # Get the standard deviation for the ContactScore
    ContactScore_stddev = np.array([ np.std(ContactScores[:,i]) for i in range(0,99) ])
    
    # An array of time steps for plotting the x axis
    time_sequence = np.linspace(0,10,99)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                        figsize = (13,4))
    fig.suptitle(f'Results for {pose_name}, {n_reps} repeats')
    
    ax1.set_title('CompScore')
    ax1.plot(time_sequence,averagedCompScore, color='blue')
    # Visualise the standard deviation of CompScore at each frame
    ax1.fill_between(time_sequence, averagedCompScore-CompScore_stddev,
                     averagedCompScore+CompScore_stddev, 
                     color='blue', alpha=0.3, lw=0)
    ax1.set_xlabel('time(ns)')
    ax1.set_ylabel('CompScore')
    ax1.set_ylim(-6,6)

    ax2.set_title('PoseScore')
    ax2.plot(time_sequence, averagedPoseScore,color='darkorange')
    ax2.fill_between(time_sequence, averagedPoseScore-PoseScore_stddev,
                     averagedPoseScore+PoseScore_stddev, 
                     color='darkorange', alpha=0.3, lw=0)
    ax2.set_xlabel('time(ns)')
    ax2.set_ylabel('PoseScore')
    ax2.set_ylim(0,5)

    ax3.set_title('ContactScore')
    ax3.plot(time_sequence, averagedContactScore,color='green')
    ax3.fill_between(time_sequence, averagedContactScore-ContactScore_stddev,
                     averagedContactScore+ContactScore_stddev, 
                     color='green', alpha=0.3, lw=0)
    ax3.set_xlabel('time(ns)')
    ax3.set_ylabel('ContactScore')
    ax3.set_ylim(-0.1,1.1)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150)
    plt.show()

    return None
