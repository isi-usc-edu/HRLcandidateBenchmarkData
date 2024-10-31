from pyscf import gto, scf

if __name__ == "__main__":

    mol = gto.M(atom='Li 0 0 0; Be 0 0 1.6', spin=1, basis='sto3g')
    mf = scf.RHF(mol)
    mf.chkfile = 'libe.chk'
    mf.run()
    print('E(HF) = %s' % mf.e_tot)