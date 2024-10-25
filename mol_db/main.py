import glob as glob

def load_chk(chk_file):
    try:
        mol = chkfile.load_mol(chk_file)
        mf = scf.ROHF(mol)
        scf_result_dic = chkfile.load(chk_file, 'scf')
        mf.__dict__.update(scf_result_dic)
    except OSError as err:
        logger.error(f"Caught OSError ({chk_file}): {err}")
        return None,None,None,None,None
    except KeyError as err:
        logger.error(f"Caught OSError ({chk_file}): {err}")
        return None,None,None,None,None
    
    nelec = mol.nelectron
    spin = mol.spin

    # H1 = Tne + Vnn
    hcore = mf.get_hcore()

    N_orbs = mol.nao_nr()
    
    # H2 = Vee
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)
    
    return mol, mf, hcore, N_orbs, eri_4d

if __name__ == '__main__':
    files = glob.glob('chks*/*')
    print(len(files))
