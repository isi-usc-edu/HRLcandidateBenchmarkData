"""
Created on Wed Sep 29 10:30:35 2021

@author: lc719e
"""

import qiskit
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)

from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ElectronicDipoleMoment,
    ParticleNumber,
    AngularMomentum,
    Magnetization,
)

from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis

import numpy as np

from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.aqua.operators import WeightedPauliOperator




class Hamiltonian_Reconstruction(object):
    """
    Converting active space NO into qubit rep for execution on QC
    """
    def __init__(self, mo_ints, gamma_point: bool = None):
        self.mo_ints = mo_ints
        self.gamma_point = gamma_point 
        particle_number = ParticleNumber(num_spin_orbitals = 2*mo_ints['no'],
                                        num_particles = (mo_ints['na'],mo_ints['nb']), )
        print('particle number:', particle_number)

    def int_coeffs_getter(self):
        mo_ints = self.mo_ints
        h1 = mo_ints['h1'][0]
        h2 = mo_ints['h2'][0][0]
        eri = np.einsum('gpr,gqs->prsq',h2,h2) 
        
        h1_real = np.real(h1)
        h1_imag = np.imag(h1)
        low_values_h1 = np.abs(h1_imag) < 1e-10  # Where values are low
        if not self.gamma_point:
            h1_imag[low_values_h1] = 0
        h1_mod = h1_real + 1j*h1_imag

        eri_real = np.real(eri)
        eri_imag = np.imag(eri)
        low_values_eri = np.abs(eri_imag) < 1e-10  # Where values are low
        if not self.gamma_point:
            eri_imag[low_values_eri] = 0
        eri_mod = eri_real + 1j*eri_imag
                        
        one_body_ints = OneBodyElectronicIntegrals(
            ElectronicBasis.MO,
            (
                h1_mod, #alpha integrals
                None             #beta integrals
            ),
        )
        
        two_body_ints = TwoBodyElectronicIntegrals(
            ElectronicBasis.MO,
            (
                eri_mod, #alpha-alpha
                None, #alpha-beta
                None, #beta-beta
                None #beta-alpha
            ),
        )

        return one_body_ints, two_body_ints 

    def fermionic_hamiltonian_getter(self):
        one_body_ints, two_body_ints = self.int_coeffs_getter() 
        electronic_energy = ElectronicEnergy(
            [one_body_ints, two_body_ints])
        
        second_quant_fermionic_hamiltonian = electronic_energy.second_q_ops()[0]  # here, output length is always 1
        return second_quant_fermionic_hamiltonian

    def qubit_hamiltonian_getter(self, qubit_mapping: str = None):
        if qubit_mapping == 'JW':
            mapper = JordanWignerMapper()
            two_qubit_reduction = False 
        else:
            mapper =  ParityMapper()
            two_qubit_reduction = True 
        mo_ints = self.mo_ints
        second_quant_fermionic_hamiltonian = self.fermionic_hamiltonian_getter() 
        qubit_converter_P = QubitConverter(mapper, two_qubit_reduction)
        qubit_hamiltonian = qubit_converter_P.convert(second_quant_fermionic_hamiltonian, 
                                                num_particles=(mo_ints['na'],mo_ints['nb']))
    
        return qubit_hamiltonian     