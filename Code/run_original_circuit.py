import warnings

# Filter the Qiskit warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, qasm3
from qiskit.circuit.library import RealAmplitudes, PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.synthesis import SuzukiTrotter
# from qiskit_nature.second_q.drivers import PySCFDriver
# from qiskit_nature.second_q.problems import ElectronicStructureProblem
# from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitMapper
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import StateTomography

# from qiskit_ibm_runtime import
import matplotlib.pyplot as plt
import time
import json

class QuantumCompressor:
    def __init__(self, num_qubits, input_circuit):
        self.num_qubits = num_qubits
        self.input_circuit = input_circuit
        self.compression_circuit = self.compression_circuit()
        self.qnn = self.setup_qnn()
        self.opt = COBYLA(maxiter=200)

    def ansatz(self):
        return RealAmplitudes(self.num_qubits, reps=15)

    def compression_circuit(self):
        qr = QuantumRegister(2 * self.num_qubits + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Apply input circuit
        circuit.compose(self.input_circuit, range(self.num_qubits), inplace=True)

        # Apply ansatz (encoder)
        circuit.compose(self.ansatz(), range(self.num_qubits), inplace=True)

        # Second set of n qubits remains in |0> state (Identity Circuit)

        circuit.barrier()

        # SWAP test
        auxiliary_qubit = 2 * self.num_qubits
        circuit.h(auxiliary_qubit)
        for i in range(self.num_qubits):
            circuit.cswap(auxiliary_qubit, i, i + self.num_qubits)
        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])

        return circuit

    def setup_qnn(self):
        def identity_interpret(x):
            return x

        return SamplerQNN(
            circuit=self.compression_circuit,
            input_params=[],
            weight_params=self.compression_circuit.parameters,
            interpret=identity_interpret,
            output_shape=2,
        )

    def cost_func(self, params_values):
        probabilities = self.qnn.forward([], params_values)
        return 1 - probabilities[0, 0]

    def train(self):
        initial_point = algorithm_globals.random.random(self.compression_circuit.num_parameters)
        start = time.time()
        opt_result = self.opt.minimize(self.cost_func, initial_point)
        elapsed = time.time() - start
        print(f"Training completed in {elapsed:.2f} seconds")
        return opt_result

    def get_compressed_circuit(self, opt_result, backend=AerSimulator()):
        compressed_circuit = QuantumCircuit(self.num_qubits)
        ansatz = self.ansatz()
        ansatz = ansatz.assign_parameters(opt_result.x)
        compressed_circuit.compose(ansatz, inplace=True)
        compressed_transpiled = transpile(compressed_circuit, backend=backend)
        # compressed_circuit.decompose(reps=4)
        print(compressed_transpiled.draw())
        compressed_circuit_depth = compressed_transpiled.depth()
        print(f"The compressed circuit depth is {compressed_circuit_depth}")
        return compressed_transpiled

    @staticmethod
    def save_circuit(circuit, filename, title):
        plt.figure(figsize=(20, 10))
        circuit.draw('mpl', style="clifford", fold=20)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def run_compression(self):
        os.makedirs("circuit_images", exist_ok=True)

        # Train the model and get the optimized result
        opt_result = self.train()

        # Get the compressed circuit
        compressed_circuit = self.get_compressed_circuit(opt_result)

        # Calculate fidelity
        input_state = Statevector(self.input_circuit).data
        compressed_state = Statevector(compressed_circuit).data
        fidelity = np.abs(np.dot(input_state.conj(), compressed_state)) ** 2

        print(f"Compression completed. Final fidelity: {fidelity:.6f}")

        return compressed_circuit, fidelity

    def save_compressed_circuit(self, compressed_circuit, filename='compressed_circuit.json'):
        """
        Save the compressed circuit to a JSON file using OpenQASM 3 format.
        """
        try:
            qasm3_str = qasm3.dumps(compressed_circuit)

            circuit_data = {
                'qasm3': qasm3_str,
                'num_qubits': self.num_qubits
            }

            # 保存為 JSON 文件
            with open(filename, 'w') as f:
                json.dump(circuit_data, f, indent=2)

            print(f"Compressed circuit saved to {filename} in OpenQASM 3 format")

        except qasm3.QASM3ExporterError as e:
            print(f"Error exporting to OpenQASM 3: {e}")
            # 如果 OpenQASM 3 導出失敗，退回到使用舊版 QASM
            qasm_str = compressed_circuit.qasm()
            circuit_data = {
                'qasm': qasm_str,
                'num_qubits': self.num_qubits
            }
            with open(filename, 'w') as f:
                json.dump(circuit_data, f, indent=2)
            print(f"Compressed circuit saved to {filename} in original QASM format due to OpenQASM 3 export failure")

    def load_compressed_circuit(self, filename='compressed_circuit.json'):
        """
        Load a compressed circuit from a JSON file, supporting both OpenQASM 3 and original QASM formats.
        """
        with open(filename, 'r') as f:
            circuit_data = json.load(f)

        if 'qasm3' in circuit_data:
            # 如果文件包含 OpenQASM 3 格式的數據
            try:
                circuit = qasm3.loads(circuit_data['qasm3'])
                print(f"Circuit loaded from {filename} using OpenQASM 3 format")
            except qasm3.QASM3ImporterError as e:
                print(f"Error importing OpenQASM 3: {e}")
                print("Attempting to load using original QASM format...")
                circuit = QuantumCircuit.from_qasm_str(circuit_data['qasm'])
                print(f"Circuit loaded from {filename} using original QASM format")
        else:
            # 如果文件只包含原始 QASM 格式的數據
            circuit = QuantumCircuit.from_qasm_str(circuit_data['qasm'])
            print(f"Circuit loaded from {filename} using original QASM format")

        return circuit

    def run_on_real_device(self, circuit, backend_name, shots=1024):
        """
        Run the circuit on a real quantum device.
        """
        # Load IBMQ account
        print("Ready to pending...")
        service = QiskitRuntimeService()

        # Get the provider and the backend
        backend = service.get_backend(backend_name)
        print(f"Get backend: {backend_name}.")

        # Transpile the circuit for the specific backend
        print("Transpiling ...")
        transpiled_circuit = transpile(circuit, backend=backend)

        # Run the circuit on the quantum device
        print("Pending...")
        qstexp1 = StateTomography(transpiled_circuit)
        qstdata1 = qstexp1.run(backend, shots=1000).block_for_results()
        state_result_1 = qstdata1.analysis_results("state")
        quantum_state_1 = state_result_1.value
        job = backend.run(transpiled_circuit, shots=shots)
        print(f"Job ID: {job.job_id()}")
        print("Running on quantum device. This may take a while...")

        # Wait for the job to complete
        print("Waiting result...")
        result = job.result()
        print("Job completed.")

        return result

    def run_compression(self):
        os.makedirs("circuit_images", exist_ok=True)

        # Train the model and get the optimized result
        opt_result = self.train()

        # Get the compressed circuit
        compressed_circuit = self.get_compressed_circuit(opt_result)

        # Calculate fidelity
        input_state = Statevector(self.input_circuit).data
        compressed_state = Statevector(compressed_circuit).data
        fidelity = np.abs(np.dot(input_state.conj(), compressed_state)) ** 2

        print(f"Compression completed. Final fidelity: {fidelity:.6f}")

        # Save the compressed circuit
        self.save_compressed_circuit(compressed_circuit)

        return compressed_circuit, fidelity

def run_saved_circuit_on_real_device(filename, backend_name, shots):
    """
    Load a saved compressed circuit and run it on a real quantum device.
    """

    # Create a QuantumCompressor instance (we need this to use its methods)
    dummy_circuit = QuantumCircuit(1)  # Dummy circuit, not actually used
    compressor = QuantumCompressor(num_qubits=1, input_circuit=dummy_circuit)

    # Load the compressed circuit
    print("Loading compressed circuit...")
    loaded_circuit = compressor.load_compressed_circuit(filename)

    # Run the loaded circuit on the real device
    print(f"Pending to {backend_name}...")
    # loaded_circuit.measure_all()
    
    result = compressor.run_on_real_device(loaded_circuit, backend_name, shots)

    return result

# Main execution
if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.get_backend('ibm_kawasaki')  # Choose an available backend
    print('Connected to IBM provider.')
    result = run_saved_circuit_on_real_device(
        filename='original_circuit_10.json',
        backend_name='ibm_kawasaki',
        shots=1024
    )
    print(result)
