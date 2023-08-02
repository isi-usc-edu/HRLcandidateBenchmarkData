from pyscf import gto, scf
from time import perf_counter
import numpy as np

def coords(n=60):
    mol = gto.M()
    mol.atom = 'C' + str(n) + '.xyz'
    mol.build()
    coords = mol.atom_coords(unit='Angstrom')
    
    new_coords  = coords
    xyz_str=""""""
    for xyz in new_coords:
        xyz_str += "C  " + str(xyz[0]) + "  " + str(xyz[1]) + "  " + str(xyz[2]) + "\n"
    
    bond_len = 1.465  # Angstroms
    bond_angle = np.deg2rad(59.9)

    x = bond_len * np.sin(bond_angle / 2)
    y = bond_len * np.cos(bond_angle / 2)

    o3 = [[x, -y / 2, 0],[-x, -y / 2, 0], [0, y / 2, 0]]
    
    for i in range(3):
        xyz_str += "O " + str(o3[i][0]) + "  " + str(o3[i][1]) + "  " + str(o3[i][2]) + "\n"
    
    return xyz_str[:-1]

if __name__ == '__main__':
    
    mol = gto.M(atom=coords(), basis = 'ccpvtz')
    mf = scf.RHF(mol)
    tic = perf_counter()
    e = mf.kernel()
    toc = perf_counter()
    print("Compute time: " + str(toc-tic))
    
    
