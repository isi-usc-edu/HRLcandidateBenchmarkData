import numpy
import scipy.linalg
from functools import reduce
import csv
from pyscf import gto, scf, mp, cc, mcscf

def make_natural_orbitals(method_obj):
    """Make natural orbitals from a general PySCF method object.
    See Eqn. (1) in Keller et al. [DOI:10.1063/1.4922352] for details.

    Args:
        method_obj : Any PySCF method that has the function `make_rdm1` with kwarg
            `ao_repr`. This object can be a restricted OR unrestricted method.

    Returns:
        noons : A 1-D array of the natural orbital occupations (NOONs).
        natorbs : A set of natural orbitals made from method_obj.
    """
    mf = method_obj
    if hasattr(method_obj, "_scf"):
        mf = method_obj._scf
    rdm1 = method_obj.make_rdm1(ao_repr=True)
    S = mf.get_ovlp()

    # Slight difference for restricted vs. unrestricted case
    if isinstance(rdm1, tuple):
        Dm = rdm1[0] + rdm1[1]
    elif isinstance(rdm1, numpy.ndarray):
        if numpy.ndim(rdm1) == 3:
            Dm = rdm1[0] + rdm1[1]
        elif numpy.ndim(rdm1) == 2:
            Dm = rdm1
        else:
            raise ValueError(
                "rdm1 passed to is a numpy array," +
                "but it has the wrong number of dimensions: {}".format(numpy.ndim(rdm1))
            )
    else:
        raise ValueError(
            "\n\tThe rdm1 generated by method_obj.make_rdm1() was a {}."
            "\n\tThis type is not supported, please select a different method and/or "
            "open an issue at https://github.com/pyscf/pyscf/issues".format(type(rdm1))
        )

    # Diagonalize the DM in AO (using Eqn. (1) referenced above)
    A = reduce(numpy.dot, (S, Dm, S))
    w, v = scipy.linalg.eigh(A, b=S)

    # Flip NOONs (and NOs) since they're in increasing order
    noons = numpy.flip(w)
    natorbs = numpy.flip(v, axis=1)

    return noons, natorbs

def make_natural_orbitals_uhf(method_obj):
    """Make natural orbitals from a general PySCF method object for alpha and beta spins.
    See Eqn. (1) in Keller et al. [DOI:10.1063/1.4922352] for details.

    Args:
        method_obj : Any PySCF method that has the function `make_rdm1` with kwarg
            `ao_repr`. This object can be a restricted OR unrestricted method.

    Returns:
        noons_alpha : A 1-D array of the natural orbital occupations (NOONs) for alpha spin.
        noons_beta : A 1-D array of the natural orbital occupations (NOONs) for beta spin.
        natorbs_alpha : A set of natural orbitals made from method_obj for alpha spin.
        natorbs_beta : A set of natural orbitals made from method_obj for beta spin.
    """
    mf = method_obj
    if hasattr(method_obj, "_scf"):
        mf = method_obj._scf
    rdm1 = method_obj.make_rdm1(ao_repr=True)
    S = mf.get_ovlp()

    if isinstance(rdm1, tuple):
        Dm_alpha, Dm_beta = rdm1
    elif isinstance(rdm1, numpy.ndarray):
        if numpy.ndim(rdm1) == 3:
            Dm_alpha, Dm_beta = rdm1
        elif numpy.ndim(rdm1) == 2:
            raise ValueError(
                "rdm1 passed to is a numpy array with 2 dimensions," +
                "but it should be a tuple or an array with 3 dimensions for unrestricted methods."
            )
        else:
            raise ValueError(
                "rdm1 passed to is a numpy array," +
                "but it has the wrong number of dimensions: {}".format(numpy.ndim(rdm1)))
    else:
        raise ValueError(
            "\n\tThe rdm1 generated by method_obj.make_rdm1() was a {}."
            "\n\tThis type is not supported, please select a different method and/or "
            "open an issue at https://github.com/pyscf/pyscf/issues".format(type(rdm1))
        )

    # Diagonalize the DM in AO for alpha spin
    A_alpha = reduce(numpy.dot, (S, Dm_alpha, S))
    w_alpha, v_alpha = scipy.linalg.eigh(A_alpha, b=S)

    # Diagonalize the DM in AO for beta spin
    A_beta = reduce(numpy.dot, (S, Dm_beta, S))
    w_beta, v_beta = scipy.linalg.eigh(A_beta, b=S)

    # Flip NOONs (and NOs) since they're in increasing order
    noons_alpha = numpy.flip(w_alpha)
    natorbs_alpha = numpy.flip(v_alpha, axis=1)

    noons_beta = numpy.flip(w_beta)
    natorbs_beta = numpy.flip(v_beta, axis=1)

    return noons_alpha, noons_beta, natorbs_alpha, natorbs_beta


# Define the geometry and spin data as a dictionary
geometry_data = {
    "c_h4": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 0.0000000000; H 1.1849715779 -1.1849715779 -1.1849715779; H -1.1849715779 1.1849715779 -1.1849715779; H -1.1849715779 -1.1849715779 1.1849715779; H 1.1849715779 1.1849715779 1.1849715779"},
    "n_h3": {"spin": 0, "atom": "N 0.0000000000 0.0000000000 0.2149752284; H -0.8857051218 1.5340862715 -0.5016466607; H -0.8857051218 -1.5340862715 -0.5016466607; H 1.7714102436 0.0000000000 -0.5016466607"},
    "h2_o": {"spin": 0, "atom": "H -1.4308249289 0.0000000000 -0.8863003855; H 1.4308249289 0.0000000000 -0.8863003855; O 0.0000000000 0.0000000000 0.2215703721"},
    "si_h2_singlet": {"spin": 0, "atom": "SI 0.0000000000 0.0000000000 0.2474879640; H -2.0687699662 0.0000000000 -1.7324195273; H 2.0687699662 0.0000000000 -1.7324195273"},
    "si_h2_triplet": {"spin": 2, "atom": "SI 0.0000000000 0.0000000000 0.1786148217; H -2.3899151028 0.0000000000 -1.2503037516; H 2.3899151028 0.0000000000 -1.2503037516"},
    "si_h3": {"spin": 1, "atom": "SI 0.0000000000 0.0000000000 0.1496228345; H -1.3303500878 2.3042339440 -0.6982386347; H -1.3303500878 -2.3042339440 -0.6982386347; H 2.6607001756 0.0000000000 -0.6982386347"},
    "si_h4": {"spin": 0, "atom": "SI 0.0000000000 0.0000000000 0.0000000000; H 1.6111331340 -1.6111331340 -1.6111331340; H -1.6111331340 1.6111331340 -1.6111331340; H -1.6111331340 -1.6111331340 1.6111331340; H 1.6111331340 1.6111331340 1.6111331340"},
    "p_h2": {"spin": 1, "atom": "P 0.0000000000 0.0000000000 -0.2185600386; H -1.9277661719 0.0000000000 1.6391993444; H 1.9277661719 0.0000000000 1.6391993444"},
    "p_h3": {"spin": 0, "atom": "P 0.0000000000 0.0000000000 0.2409778580; H -1.1208437265 1.9413582816 -1.2049081870; H -1.1208437265 -1.9413582816 -1.2049081870; H 2.2416874529 0.0000000000 -1.2049081870"},
    "h2_s": {"spin": 0, "atom": "H -1.8156865235 0.0000000000 -1.5585704056; H 1.8156865235 0.0000000000 -1.5585704056; S 0.0000000000 0.0000000000 0.1948307493"},
    "h_cl": {"spin": 0, "atom": "H 0.0000000000 0.0000000000 -2.2744741988; CL 0.0000000000 0.0000000000 0.1337925999"},
    "c2_h2": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 -1.1361032638; C 0.0000000000 0.0000000000 1.1361032638; H 0.0000000000 0.0000000000 -3.1439371258; H 0.0000000000 0.0000000000 3.1439371258"},
    "c2_h4": {"spin": 0, "atom": "C 0.000000000 0.000000000 0.000000000; C 0.000000000 0.000000000 2.508023735; H 1.740983898 0.000000000 -1.085783723; H -1.740983898 0.000000000 -1.085783723; H 1.740983898 0.000000000 3.593807457; H -1.740983898 0.000000000 3.593807457"},
    "c2_h6": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 -1.4386483945; C 0.0000000000 0.0000000000 1.4386483945; H -1.6617535925 -0.9594138840 2.1829169747; H 1.6617535925 -0.9594138840 2.1829169747; H 0.0000000000 -1.9188277679 -2.1829169747; H 1.6617535925 0.9594138840 -2.1829169747; H -1.6617535925 0.9594138840 -2.1829169747; H 0.0000000000 1.9188277679 2.1829169747"},
    "h_c_n": {"spin": 0, "atom": "H 0.0000000000 0.0000000000 -2.8401825705; C 0.0000000000 0.0000000000 -0.8276999826; N 0.0000000000 0.0000000000 1.3516076155"},
    "c_o": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 1.0752540870; O 0.0000000000 0.0000000000 -1.0752540870"},
    "h_c_o": {"spin": 1, "atom": "H 1.767383372 0.000000000 -1.193700313; C 0.000000000 0.000000000 0.000000000; O 0.000000000 0.000000000 2.222716663"},
    "h2_c_o": {"spin": 0, "atom": "H -1.7690669834 0.0000000000 -2.0934384492; H 1.7690669834 0.0000000000 -2.0934384492; C 0.0000000000 0.0000000000 -1.0008555749; O 0.0000000000 0.0000000000 1.2739965691"},
    "h3_c_o_h": {"spin": 0, "atom": "H -2.0540490007 1.8442554010 0.0000000000; H 1.6262830672 -1.9975140683 0.0000000000; H 0.8279739929 2.0250190301 -1.6809849654; H 0.8279739929 2.0250190301 1.6809849654; C -0.0877267495 1.2530187210 0.0000000000; O -0.0877267495 -1.4268621735 0.0000000000"},
    "n2": {"spin": 0, "atom": "N 0.0000000000 0.0000000000 -1.0370816221; N 0.0000000000 0.0000000000 1.0370816221"},
    "n2_h4": {"spin": 0, "atom": "N 0.1122505457 1.3501196282 -0.0803855930; N -0.1122505457 -1.3501196282 -0.0803855930; H 0.5895561278 2.0205227427 1.6458351020; H -0.5895561278 -2.0205227427 1.6458351020; H -1.5986022854 2.0625696108 -0.5287677360; H 1.5986022854 -2.0625696108 -0.5287677360"},
    "o2": {"spin": 2, "atom": "O 0.0000000000 0.0000000000 -1.1409220651; O 0.0000000000 0.0000000000 1.1409220651"},
    "h2_o2": {"spin": 0, "atom": "H -1.6869016975 -1.4931291947 0.8801398788; H 1.6869016975 1.4931291947 0.8801398788; O -1.3709962041 0.0000000000 -0.1100198470; O 1.3709962041 0.0000000000 -0.1100198470"},
    "f2": {"spin": 0, "atom": "F 0.0000000000 0.0000000000 -1.3339575747; F 0.0000000000 0.0000000000 1.3339575747"},
    "c_o2": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 0.0000000000; O 0.0000000000 0.0000000000 -2.1920821458; O 0.0000000000 0.0000000000 2.1920821458"},
    "na2": {"spin": 0, "atom": "NA 0.0000000000 0.0000000000 -2.9091386718; NA 0.0000000000 0.0000000000 2.9091386718"},
    "si2": {"spin": 2, "atom": "SI 0.0000000000 0.0000000000 -2.1221622842; SI 0.0000000000 0.0000000000 2.1221622842"},
    "p2": {"spin": 0, "atom": "P 0.0000000000 0.0000000000 -1.7890035926; P 0.0000000000 0.0000000000 1.7890035926"},
    "s2": {"spin": 2, "atom": "S 0.0000000000 0.0000000000 -1.7850351680; S 0.0000000000 0.0000000000 1.7850351680"},
    "cl2": {"spin": 0, "atom": "CL 0.0000000000 0.0000000000 -1.8782931455; CL 0.0000000000 0.0000000000 1.8782931455"},
    "na_cl": {"spin": 0, "atom": "NA 0.0000000000 0.0000000000 0.0000000000; CL 0.0000000000 0.0000000000 4.4612651118"},
    "si_o": {"spin": 0, "atom": "SI 0.0000000000 0.0000000000 1.0376296426; O 0.0000000000 0.0000000000 -1.8158565989"},
    "c_s": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 1.4513095586; S 0.0000000000 0.0000000000 -1.4513095586"},
    "s_o": {"spin": 2, "atom": "S 0.0000000000 0.0000000000 0.9331466927; O 0.0000000000 0.0000000000 -1.8662933855"},
    "cl_o": {"spin": 1, "atom": "CL 0.0000000000 0.0000000000 1.4815451744; O 0.0000000000 0.0000000000 -1.4815451744"},
    "cl_f": {"spin": 0, "atom": "CL 0.0000000000 0.0000000000 1.5382369540; F 0.0000000000 0.0000000000 -1.5382369540"},
    "si2_h6": {"spin": 0, "atom": "SI 0.0000000000 0.0000000000 -2.2115557721; SI 0.0000000000 0.0000000000 2.2115557721; H -2.2717107717 -1.3115728255 3.1851066961; H 2.2717107717 -1.3115728255 3.1851066961; H 0.0000000000 -2.6231456511 -3.1851066961; H 2.2717107717 1.3115728255 -3.1851066961; H -2.2717107717 1.3115728255 -3.1851066961; H 0.0000000000 2.6231456511 3.1851066961"},
    "c_h3_cl": {"spin": 0, "atom": "C 0.0000000000 0.0000000000 -2.1213308048; H -0.9712407341 1.6822382978 -2.7675093782; H -0.9712407341 -1.6822382978 -2.7675093782; H 1.9424814681 0.0000000000 -2.7675093782; CL 0.0000000000 0.0000000000 1.2370902206"},
    "h3_c_s_h": {"spin": 0, "atom": "H 2.4151850856 -1.5678281732 0.0000000000; H -2.0620727773 2.7522536203 0.0000000000; H 0.8159421075 2.9203504161 -1.6837534140; H 0.8159421075 2.9203504161 1.6837534140; C -0.0902268570 2.1733908660 0.0000000000; S -0.0902268570 -1.2540920853 0.0000000000"},
    "h_o_cl": {"spin": 0, "atom": "H -1.7072351491 2.4777898178 0.0000000000; O 0.0682946972 2.0711396825 0.0000000000; CL 0.0682946972 -1.1204185381 0.0000000000"},
    "s_o2": {"spin": 0, "atom": "S 0.0000000000 0.0000000000 0.0000000000; O 0.0000000000 -2.3377800194 1.3634373001; O 0.0000000000 2.3377800194 1.3634373001"},
}


if __name__ == "__main__":
    # Open a CSV file for writing the results
    with open("calculation_results.csv", mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write headers to the CSV file
        csv_writer.writerow(["Geometry Name", "CASCI Energy", "CCSD Full Energy", "CCSD Frozen Energy"])
        
        # Loop over all geometries in the dictionary
        for geometry_name, geometry in geometry_data.items():
            print(f"\nRunning calculations for {geometry_name}...")

            # Get the geometry and spin for the specified name
            atoms = geometry["atom"]
            spin = geometry["spin"]

            mol = gto.Mole()
            mol.unit = 'bohr'
            mol.atom = atoms
            mol.basis = 'cc-pvdz'
            mol.spin = spin
            mol.build()

            if mol.spin == 0:
                # Closed-shell calculation
                # Perform SCF calculation
                mf = scf.RHF(mol)
                # Set convergence options
                mf.conv_tol = 1e-8  # Tight convergence threshold
                mf.max_cycle = 1000  # Increase maximum number of SCF cycles
                mf.init_guess = 'mix'  # Use mixed initial guess
                mf.level_shift = 0.1  # Level shift to improve convergence for difficult cases
                mf.diis_space = 8  # Increase DIIS space to improve convergence
                mf.verbose = 4
                mf = scf.newton(mf)
                mf.kernel()

                # Internal stability analysis
                mo1, _, stable, _ = mf.stability(return_status=True)
                if not stable:
                    print('Initial wavefunction is internally unstable. Attempting to optimize...')
                    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
                    mf = mf.run(dm1)
                    mo1, _, stable, _ = mf.stability(return_status=True)
                    if not stable:
                        print('Wavefunction remains internally unstable after optimization attempts.')

                # External stability analysis
                mo2, _, stable_ext, _ = mf.stability(external=True, return_status=True)
                if not stable_ext:
                    print('Initial wavefunction is externally unstable. Attempting to optimize...')
                    dm2 = mf.make_rdm1(mo2, mf.mo_occ)
                    mf = mf.run(dm2)
                    mo2, _, stable_ext, _ = mf.stability(external=True, return_status=True)
                    if not stable_ext:
                        print('Wavefunction remains externally unstable after optimization attempts.')

                # Perform MP2 calculation
                mp2 = mp.MP2(mf)
                mp2.kernel()

                # Generate natural orbitals
                noons, natorbs = make_natural_orbitals(mp2)

                # Print the natural orbital occupations with orbital number and sum of occupations
                print("Natural Orbital Occupations:")
                for i, occ in enumerate(noons):
                    print(f"Orbital {i}: {occ}")

                # Select natural orbitals with occupation between 1.0 and 1.99
                selected_orbitals = [i for i, occ in enumerate(noons) if 1.0 <= occ <= 1.99]

                print("\nSelected Natural Orbital Indices with Occupation between 1.0 and 1.99:")
                print(selected_orbitals)

                # Print the maximum index in total orbitals
                max_index_total = len(noons) - 1
                print("\nMaximum Index in Total Orbitals: ", max_index_total)

                # Print the number of electrons for the system
                num_electrons = mol.nelectron
                print("\nNumber of Electrons in the System: ", num_electrons)

                # Handle empty selected_orbitals
                if not selected_orbitals:
                    print("\nNo selected orbitals found. Skipping to next molecule.")
                    continue

                # Count one electron for each index before the combined list for the inactive space
                inactive_electrons = len([i for i in range(min(selected_orbitals))]) * 2
                print("\nNumber of Electrons in Inactive Space: ", inactive_electrons)

                # Count active electrons
                active_electrons = num_electrons - inactive_electrons
                print("\nNumber of Active Electrons: ", active_electrons)

                # Update selected_orbitals to match the number of active electrons
                while len(selected_orbitals) < active_electrons:
                    next_index = max(selected_orbitals) + 1
                    if next_index <= max_index_total:
                        selected_orbitals.append(next_index)
                    else:
                        break

                print("\nUpdated Selected Natural Orbital Indices to match Active Electrons:")
                print(sorted(selected_orbitals))

                # Update frozen_orbitals to include all orbitals excluding active orbitals
                frozen_orbitals = [i for i in range(max_index_total + 1) if i not in selected_orbitals]

                print("\nUpdated Frozen Orbitals:")
                print(frozen_orbitals)

                # Set the coefficients to the natural orbitals
                mf.mo_coeff = natorbs

                # Generate a 1-based index list for CASCI active space
                casci_indices = [i + 1 for i in selected_orbitals]
                print("\n1-Based CASCI Indices:")
                print(casci_indices)

                # Perform frozen CCSD calculation
                print("Frozen CCSD Energies")
                ccsd_frozen = cc.CCSD(mf, frozen=frozen_orbitals)
                ccsd_frozen.verbose = 5
                ccsd_frozen.max_cycle = 500  # Increase the maximum number of CCSD cycles

                # Use DIIS to help with convergence
                ccsd_frozen.diis_start_cycle = 4  # Start DIIS after 4 cycles
                ccsd_frozen.diis_space = 8  # Use a DIIS space of 8

                # Adjust the convergence threshold
                ccsd_frozen.conv_tol = 1e-6  # Convergence threshold for energy

                ccsd_frozen.kernel()
                ccsd_frozen_energy = ccsd_frozen.e_tot  # Store the energy

                # Perform full CCSD calculation
                print("Full CCSD Energies")
                ccsd_full = cc.CCSD(mf)
                ccsd_full.verbose = 5
                ccsd_full.max_cycle = 500  # Increase the maximum number of CCSD cycles

                # Use DIIS to help with convergence
                ccsd_full.diis_start_cycle = 4  # Start DIIS after 4 cycles
                ccsd_full.diis_space = 12  # Use a DIIS space of 8

                # Adjust the convergence threshold
                ccsd_full.conv_tol = 1e-6  # Convergence threshold for energy
                ccsd_full.kernel()
                ccsd_full_energy = ccsd_full.e_tot  # Store the energy

                nelec = active_electrons  # Number of active electrons
                ncas = len(selected_orbitals)  # Number of active orbitals

                # Perform CASCI calculation
                mc = mcscf.CASCI(mf, ncas, nelec)
                no_orb = mc.sort_mo(casci_indices)

                mc.verbose = 4  # Set verbosity level to print more information
                mc.kernel(no_orb)
                casci_energy = mc.e_tot  # Store the CASCI energy

            else:
                # Open-shell calculation
                # Perform SCF calculation
                mf = scf.UHF(mol)
                # Set convergence options
                mf.conv_tol = 1e-8  # Tight convergence threshold
                mf.max_cycle = 1000  # Increase maximum number of SCF cycles
                mf.init_guess = 'mix'  # Use mixed initial guess
                mf.level_shift = 0.1  # Level shift to improve convergence for difficult cases
                mf.diis_space = 8  # Increase DIIS space to improve convergence
                mf.verbose = 4
                mf = scf.newton(mf)
                mf.kernel()

                # Internal stability analysis
                mo1, _, stable, _ = mf.stability(return_status=True)
                if not stable:
                    print('Initial wavefunction is internally unstable. Attempting to optimize...')
                    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
                    mf = mf.run(dm1)
                    mo1, _, stable, _ = mf.stability(return_status=True)
                    if not stable:
                        print('Wavefunction remains internally unstable after optimization attempts.')

                # External stability analysis
                mo2, _, stable_ext, _ = mf.stability(external=True, return_status=True)
                if not stable_ext:
                    print('Initial wavefunction is externally unstable. Attempting to optimize...')
                    dm2 = mf.make_rdm1(mo2, mf.mo_occ)
                    mf = mf.run(dm2)
                    mo2, _, stable_ext, _ = mf.stability(external=True, return_status=True)
                    if not stable_ext:
                        print('Wavefunction remains externally unstable after optimization attempts.')

                # Perform MP2 calculation
                mp2 = mp.UMP2(mf)
                mp2.kernel()

                # Generate natural orbitals
                noons_alpha, noons_beta, natorbs_alpha, natorbs_beta = make_natural_orbitals_uhf(mp2)

                # Print the natural orbital occupations with orbital number and sum of occupations
                print("Natural Orbital Occupations (Alpha):")
                for i, occ in enumerate(noons_alpha):
                    print(f"Orbital {i}: {occ}")

                print("\nNatural Orbital Occupations (Beta):")
                for i, occ in enumerate(noons_beta):
                    print(f"Orbital {i}: {occ}")

                # Calculate and print the sum of alpha and beta occupations
                sum_noons = noons_alpha + noons_beta
                print("\nSum of Natural Orbital Occupations (Alpha + Beta):")
                for i, occ in enumerate(sum_noons):
                    print(f"Orbital {i}: {occ}")

                # Select natural orbitals with occupation between 0.5 and 0.99
                selected_orbitals_alpha = [i for i, occ in enumerate(noons_alpha) if 0.5 <= occ <= 0.995]
                selected_orbitals_beta = [i for i, occ in enumerate(noons_beta) if 0.5 <= occ <= 0.995]

                # Combine indices, remove duplicates, and sort
                combined_indices = sorted(set(selected_orbitals_alpha + selected_orbitals_beta))

                print("\nCombined Natural Orbital Indices (Alpha + Beta) with Occupation between 0.5 and 0.995:")
                print(combined_indices)

                # Print the maximum index in total orbitals
                max_index_total = max(len(noons_alpha), len(noons_beta)) - 1
                print("\nMaximum Index in Total Orbitals: ", max_index_total)

                # Print the number of electrons for the system
                num_electrons = mol.nelectron
                print("\nNumber of Electrons in the System: ", num_electrons)

                # Handle empty combined_indices
                if not combined_indices:
                    print("\nNo combined indices found. Skipping to next molecule.")
                    continue

                # Count one electron for each index before the combined list for the inactive space, doubling for alpha and beta
                inactive_electrons = 2 * len([i for i in range(min(combined_indices))])
                print("\nNumber of Electrons in Inactive Space: ", inactive_electrons)

                # Count active electrons
                active_electrons = num_electrons - inactive_electrons
                print("\nNumber of Active Electrons: ", active_electrons)

                # Update combined_indices to match the number of active electrons
                while len(combined_indices) < active_electrons:
                    next_index = max(combined_indices) + 1
                    if next_index <= max_index_total:
                        combined_indices.append(next_index)
                    else:
                        break

                print("\nUpdated Combined Natural Orbital Indices to match Active Electrons:")
                print(sorted(combined_indices))

                # Find the number of alpha and beta electrons in active space
                min_index_combined = min(combined_indices)
                max_index_alpha = max(selected_orbitals_alpha) if selected_orbitals_alpha else min_index_combined
                max_index_beta = max(selected_orbitals_beta) if selected_orbitals_beta else min_index_combined

                active_alpha_electrons = sum(1 for i in range(min_index_combined, max_index_alpha + 1) if i in combined_indices)
                active_beta_electrons = sum(1 for i in range(min_index_combined, max_index_beta + 1) if i in combined_indices)

                print("\nNumber of Alpha Electrons in Active Space: ", active_alpha_electrons)
                print("\nNumber of Beta Electrons in Active Space: ", active_beta_electrons)

                # Update frozen_orbitals to include all orbitals excluding active orbitals
                frozen_orbitals = [i for i in range(max_index_total + 1) if i not in combined_indices]

                print("\nUpdated Frozen Orbitals:")
                print(frozen_orbitals)

                # Set the coefficients to the natural orbitals
                mf.mo_coeff[0] = natorbs_alpha
                mf.mo_coeff[1] = natorbs_beta

                # Generate a 1-based index list for CASCI active space
                casci_indices = [i + 1 for i in combined_indices]
                print("\n1-Based CASCI Indices:")
                print(casci_indices)

                # Perform frozen CCSD calculation
                
                ccsd_frozen = cc.UCCSD(mf, frozen=frozen_orbitals)
                ccsd_frozen.verbose = 5
                ccsd_frozen.max_cycle = 500  # Increase the maximum number of CCSD cycles

                # Use DIIS to help with convergence
                ccsd_frozen.diis_start_cycle = 4  # Start DIIS after 4 cycles
                ccsd_frozen.diis_space = 12  # Use a DIIS space of 8

                # Adjust the convergence threshold
                ccsd_frozen.conv_tol = 1e-6  # Convergence threshold for energy
                ccsd_frozen.kernel()
                print("Frozen CCSD Energy:")
                ccsd_frozen_energy = ccsd_frozen.e_tot  # Store the energy

                # Perform full CCSD calculation
                
                ccsd_full = cc.UCCSD(mf)
                ccsd_full.verbose = 5
                ccsd_full.max_cycle = 500  # Increase the maximum number of CCSD cycles

                # Use DIIS to help with convergence
                ccsd_full.diis_start_cycle = 4  # Start DIIS after 4 cycles
                ccsd_full.diis_space = 8  # Use a DIIS space of 8

                # Adjust the convergence threshold
                ccsd_full.conv_tol = 1e-6  # Convergence threshold for energy
                ccsd_full.kernel()
                print("Full CCSD Energy:")
                ccsd_full_energy = ccsd_full.e_tot  # Store the energy

                nelec = (active_alpha_electrons, active_beta_electrons)  # Number of active electrons
                ncas = len(combined_indices)  # Number of active orbitals

                # Perform CASCI calculation
                mc = mcscf.UCASCI(mf, ncas, nelec)
                no_orb = mc.sort_mo(casci_indices)

                mc.verbose = 4  # Set verbosity level to print more information
                mc.kernel(no_orb)
                casci_energy = mc.e_tot  # Store the CASCI energy

            # Write the results for this geometry to the CSV file
            csv_writer.writerow([geometry_name, casci_energy, ccsd_full_energy, ccsd_frozen_energy])

            print(f"Completed calculations for {geometry_name}.\n")
