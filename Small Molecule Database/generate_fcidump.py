import glob as glob
from pyscf import scf, ao2mo
from pyscf.lib import chkfile
from pyscf.tools import fcidump
import re

def load_chk(chk_file):
    mol = chkfile.load_mol(chk_file)
    mf = scf.RHF(mol)
    scf_result_dic = chkfile.load(chk_file, 'scf')
    mf.__dict__.update(scf_result_dic)

    nelec = mol.nelectron
    spin = mol.spin

    # H1 = Tne + Vnn
    hcore = mf.get_hcore()

    N_orbs = mol.nao_nr()

    # H2 = Vee
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)

    return mol, mf, hcore, N_orbs, eri_4d, mf.mo_coeff, nelec

def extract_substring(a):
    # Use regex to find the part between the first / and the last _
    match = re.search(r'\/([^\/]*)_[^_]*$', a)
    return match.group(1) if match else None

def generate_FCIDUMP(chk_file):
    result = extract_substring(chk_file)
    mol, mf, hcore, N_orbs, eri_4d, mo_coeff, nelec = load_chk(chk_file)
    filename = './FCIDUMPS/' + result + ".FCIDUMP"

    fcidump.from_integrals(filename, hcore, eri_4d, N_orbs, nelec)

if __name__ == '__main__':
    files = glob.glob('*_chks/*.chk')

    chk_file = files[0]
    generate_FCIDUMP(chk_file)
