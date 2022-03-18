# Usage
# python example.py 01
import numpy as np
import sys
import tequila as tq
import qiskit
import pickle
filename = 'xyz/CH3CL_CL_' + sys.argv[1] + '.xyz'
active_orbitals = {"A":[21,22]}
basis = 'sto-3g'

molecule = tq.chemistry.Molecule(geometry=filename,
                                 backend='psi4',
                                 charge=-1,
                                 basis_set=basis,
                                 active_orbitals=active_orbitals,
                                 transformation="bravyi_kitaev")

# Make Hamiltonian and Ansatzes
H = molecule.make_hamiltonian()
U = molecule.make_uccsd_ansatz(initial_amplitudes="mp2")
E = tq.ExpectationValue(H=H, U=U)
vqe = tq.minimize(method="cobyla", objective=E, initial_values=0.0)

UpCCGSD1 = molecule.make_upccgsd_ansatz(order=1)
E1 = tq.ExpectationValue(H=H, U=UpCCGSD1, optimize_measurements=True)
ucc1 = tq.minimize(method="cobyla", objective=E1, initial_values=0.0)

UpCCGSD2 = molecule.make_upccgsd_ansatz(order=2)
E2 = tq.ExpectationValue(H=H, U=UpCCGSD2, optimize_measurements=True)
ucc2 = tq.minimize(method="cobyla", objective=E2, initial_values=0.0)

UpCCGSD3 = molecule.make_upccgsd_ansatz(order=3)
E3 = tq.ExpectationValue(H=H, U=UpCCGSD3, optimize_measurements=True)
ucc3 = tq.minimize(method="cobyla", objective=E3, initial_values=0.0)

UpCCGSD4 = molecule.make_upccgsd_ansatz(order=4)
E4 = tq.ExpectationValue(H=H, U=UpCCGSD4, optimize_measurements=True)
ucc4 = tq.minimize(method="cobyla", objective=E4, initial_values=0.0)

UpCCGSD5 = molecule.make_upccgsd_ansatz(order=5)
E5 = tq.ExpectationValue(H=H, U=UpCCGSD5, optimize_measurements=True)
ucc5 = tq.minimize(method="cobyla", objective=E5, initial_values=0.0)

print("VQE         = {:2.8f}".format(min(vqe.history.energies)))
print("1-UCC       = {:2.8f}".format(min(ucc1.history.energies)))
print("2-UCC       = {:2.8f}".format(min(ucc2.history.energies)))
print("3-UCC       = {:2.8f}".format(min(ucc3.history.energies)))
print("4-UCC       = {:2.8f}".format(min(ucc4.history.energies)))
print("5-UCC       = {:2.8f}".format(min(ucc5.history.energies)))