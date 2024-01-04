import numpy as np
import qiskit
from qiskit.compiler import *
from qiskit.transpiler.passes import RemoveBarriers
import re
import numpy as np

"""
Benchmark class for QASM strings. Preprocessing currently taken from QASMBench.
Would likely be better to call QASMBench for these functions in the future.
"""

small_qasm = ['wstate_n3.qasm', 'deutsch_n2_transpiled.qasm', 'adder_n10.qasm', 'vqe_uccsd_n4_transpiled.qasm', 'bell_n4_transpiled.qasm', 'vqe_n4.qasm', 'hhl_n10.qasm', 'adder_n4.qasm', 'grover_n2.qasm', 'vqe_uccsd_n4.qasm', 'grover_n2_transpiled.qasm', 'adder_n4_transpiled.qasm', 'pea_n5_transpiled.qasm', 'qft_n4.qasm', 'basis_test_n4.qasm', 'dnn_n8.qasm', 'bell_n4.qasm', 'simon_n6_transpiled.qasm', 'ising_n10.qasm', 'variational_n4_transpiled.qasm', 'vqe_uccsd_n8.qasm', 'vqe_n4_transpiled.qasm', 'sat_n7.qasm', 'qaoa_n6.qasm', 'iswap_n2_transpiled.qasm', 'bb84_n8_transpiled.qasm', 'cat_state_n4_transpiled.qasm', 'dnn_n8_transpiled.qasm', 'lpn_n5_transpiled.qasm', 'qpe_n9_transpiled.qasm', 'qaoa_n3_transpiled.qasm', 'qec_sm_n5.qasm', 'hhl_n7_transpiled.qasm', 'qpe_n9.qasm', 'quantumwalks_n2_transpiled.qasm', 'hs4_n4_transpiled.qasm', 'error_correctiond3_n5.qasm', 'dnn_n2.qasm', 'bb84_n8.qasm', 'shor_n5_transpiled.qasm', 'basis_trotter_n4_transpiled.qasm', 'inverseqft_n4.qasm', 'qrng_n4.qasm', 'basis_trotter_n4.qasm', 'wstate_n3_transpiled.qasm', 'qrng_n4_transpiled.qasm', 'qec_en_n5.qasm', 'adder_n10_transpiled.qasm', 'cat_state_n4.qasm', 'basis_change_n3_transpiled.qasm', 'shor_n5.qasm', 'qaoa_n3.qasm', 'hhl_n7.qasm', 'vqe_uccsd_n8_transpiled.qasm', 'dnn_n2_transpiled.qasm', 'hs4_n4.qasm', 'inverseqft_n4_transpiled.qasm', 'fredkin_n3_transpiled.qasm', 'deutsch_n2.qasm', 'qaoa_n6_transpiled.qasm', 'teleportation_n3_transpiled.qasm', 'ipea_n2.qasm', 'iswap_n2.qasm', 'ipea_n2_transpiled.qasm', 'hhl_n10_transpiled.qasm', 'lpn_n5.qasm', 'toffoli_n3.qasm', 'qft_n4_transpiled.qasm', 'vqe_uccsd_n6_transpiled.qasm', 'simon_n6.qasm', 'toffoli_n3_transpiled.qasm', 'fredkin_n3.qasm', 'qec_sm_n5_transpiled.qasm', 'basis_test_n4_transpiled.qasm', 'error_correctiond3_n5_transpiled.qasm', 'pea_n5.qasm', 'variational_n4.qasm', 'ising_n10_transpiled.qasm', 'teleportation_n3.qasm', 'qec_en_n5_transpiled.qasm', 'vqe_uccsd_n6.qasm', 'basis_change_n3.qasm', 'linearsolver_n3_transpiled.qasm', 'quantumwalks_n2.qasm', 'linearsolver_n3.qasm']
medium_qasm = ['multiply_n13_transpiled.qasm', 'swap_test_n25_transpiled.qasm', 'qram_n20_transpiled.qasm', 'ising_n26.qasm', 'wstate_n27_transpiled.qasm', 'sat_n11_transpiled.qasm', 'bwt_n21.qasm', 'dnn_n16_transpiled.qasm', 'cat_state_n22.qasm', 'gcm_h6.qasm', 'vqe_n24.qasm', 'bv_n14_transpiled.qasm', 'ghz_state_n23.qasm', 'wstate_n27.qasm', 'cc_n12_transpiled.qasm', 'qft_n18.qasm', 'dnn_n16.qasm', 'qram_n20.qasm', 'ising_n26_transpiled.qasm', 'seca_n11_transpiled.qasm', 'bwt_n21_transpiled.qasm', 'qec9xz_n17.qasm', 'multiplier_n15_transpiled.qasm', 'hhl_n14.qasm', 'knn_n25.qasm', 'qf21_n15_transpiled.qasm', 'square_root_n18_transpiled.qasm', 'qec9xz_n17_transpiled.qasm', 'swap_test_n25.qasm', 'bv_n19_transpiled.qasm', 'cat_state_n22_transpiled.qasm', 'multiplier_n15.qasm', 'seca_n11.qasm', 'vqe_n24_transpiled.qasm', 'knn_n25_transpiled.qasm', 'ghz_state_n23_transpiled.qasm', 'cc_n12.qasm', 'multiply_n13.qasm', 'bv_n19.qasm', 'qf21_n15.qasm', 'qft_n18_transpiled.qasm', 'factor247_n15.qasm', 'bigadder_n18_transpiled.qasm', 'bigadder_n18.qasm', 'sat_n11.qasm', 'square_root_n18.qasm', 'bv_n14.qasm']
large_qasm = ['wstate_n36_transpiled.qasm', 'cc_n301.qasm', 'wstate_n118.qasm', 'square_root_n60.qasm', 'ising_n98_transpiled.qasm', 'qugan_n39.qasm', 'wstate_n118_transpiled.qasm', 'wstate_n76.qasm', 'qugan_n111_transpiled.qasm', 'cat_n130.qasm', 'swap_test_n115.qasm', 'bwt_n57.qasm', 'ising_n66.qasm', 'qft_n29.qasm', 'swap_test_n83.qasm', 'qugan_n111.qasm', 'knn_129.qasm', 'multiplier_n400.qasm', 'swap_test_n83_transpiled.qasm', 'qugan_n395_transpiled.qasm', 'square_root_n45_transpiled.qasm', 'swap_test_n41.qasm', 'bv_n280_transpiled.qasm', 'ghz_state_n255_transpiled.qasm', 'multiplier_n75_transpiled.qasm', 'bwt_n37.qasm', 'wstate_n380.qasm', 'ghz_state_n255.qasm', 'swap_test_n41_transpiled.qasm', 'random_QAOA_angles_k3_N10000_p1.qasm', 'dnn_n51_transpiled.qasm', 'wstate_n36.qasm', 'dnn_n51.qasm', 'wstate_n76_transpiled.qasm', 'knn_n41.qasm', 'bwt_n97_transpiled.qasm', 'adder_n433.qasm', 'knn_n67_transpiled.qasm', 'bv_n30_transpiled.qasm', 'bv_n70.qasm', 'ising_n420_transpiled.qasm', 'square_root_n60_transpiled.qasm', 'knn_n31.qasm', 'cc_n301_transpiled.qasm', 'knn_129_transpiled.qasm', 'multiplier_n400_transpiled.qasm', 'qft_n29_transpiled.qasm', 'ising_n66_transpiled.qasm', 'cc_n151.qasm', 'dnn_n33_transpiled.qasm', 'swap_test_n115_transpiled.qasm', 'bwt_n37_transpiled.qasm', 'vqe_uccsd_n28_transpiled.qasm', 'swap_test_n361_transpiled.qasm', 'qft_n63.qasm', 'knn_n31_transpiled.qasm', 'multiplier_n350_transpiled.qasm', 'random_QAOA_angles_k3_N100_p100.qasm', 'ghz_n127_transpiled.qasm', 'knn_n67.qasm', 'ghz_n78.qasm', 'cc_n151_transpiled.qasm', 'ising_n98.qasm', 'bv_n70_transpiled.qasm', 'bv_n30.qasm', 'qugan_n395.qasm', 'ghz_n40_transpiled.qasm', 'adder_n28_transpiled.qasm', 'multiplier_n75.qasm', 'qft_n63_transpiled.qasm', 'cat_n35.qasm', 'qft_n320.qasm', 'adder_n433_transpiled.qasm', 'ghz_n78_transpiled.qasm', 'qft_n160.qasm', 'knn_341.qasm', 'qugan_n39_transpiled.qasm', 'adder_n118_transpiled.qasm', 'ising_n42.qasm', 'qugan_n71_transpiled.qasm', 'knn_341_transpiled.qasm', '32.qasm', 'bv_n140.qasm', 'qft_n160_transpiled.qasm', 'adder_n118.qasm', 'ghz_n40.qasm', 'ising_n34.qasm', 'qugan_n71.qasm', 'multiplier_n350.qasm', 'cat_n35_transpiled.qasm', 'bv_n140_transpiled.qasm', 'square_root_n45.qasm', 'random_QAOA_angles_k3_N1000_p1.qasm', 'bwt_n57_transpiled.qasm', 'cat_n65_transpiled.qasm', 'adder_n28.qasm', 'multiplier_n45.qasm', 'bwt_n177.qasm', 'cat_n260.qasm', 'bv_n280.qasm', 'cc_n64_transpiled.qasm', 'swap_test_n361.qasm', 'cc_n32.qasm', 'qft_n320_transpiled.qasm', 'dnn_n33.qasm', 'adder_n64.qasm', 'ghz_n127.qasm', 'cc_n64.qasm', 'wstate_n380_transpiled.qasm', 'cat_n65.qasm', 'cat_n130_transpiled.qasm', 'ising_n34_transpiled.qasm', 'ising_n420.qasm', 'bwt_n97.qasm', 'adder_n64_transpiled.qasm', 'multiplier_n45_transpiled.qasm', '100.qasm', 'cat_n260_transpiled.qasm', 'vqe_uccsd_n28.qasm', 'cc_n32_transpiled.qasm']
class Benchmark:

    def __init__(self, qasm):
        self.qasm = qasm
        # =======  Global tables and variables =========

        # Standard gates are gates defined in OpenQASM header.
        # Dictionary in {"gate name": number of standard gates inside}
        self.STANDARD_GATE_TABLE = {
            "r": 1,   # 2-Parameter rotation around Z-axis and X-axis
            "sx": 1,  # SX Gate - Square root X gate
            "u3": 1,  # 3-parameter 2-pulse single qubit gate
            "u2": 1,  # 2-parameter 1-pulse single qubit gate
            "u1": 1,  # 1-parameter 0-pulse single qubit gate
            "cx": 1,  # controlled-NOT
            "id": 1,  # idle gate(identity)
            "x": 1,  # Pauli gate: bit-flip
            "y": 1,  # Pauli gate: bit and phase flip
            "z": 1,  # Pauli gate: phase flip
            "h": 1,  # Clifford gate: Hadamard
            "s": 1,  # Clifford gate: sqrt(Z) phase gate
            "sdg": 1,  # Clifford gate: conjugate of sqrt(Z)
            "t": 1,  # C3 gate: sqrt(S) phase gate
            "tdg": 1,  # C3 gate: conjugate of sqrt(S)
            "rx": 1,  # Rotation around X-axis
            "ry": 1,  # Rotation around Y-axis
            "rz": 1,  # Rotation around Z-axis
            "c1": 1,  # Arbitrary 1-qubit gate
            "c2": 1}  # Arbitrary 2-qubit gate

        # Composition gates are gates defined in OpenQASM header.
        # Dictionary in {"gate name": number of standard gates inside}
        self.COMPOSITION_GATE_TABLE = {
            "p":1, # Phase Gate
            "cz": 3,  # Controlled-Phase
            "cy": 3,  # Controlled-Y
            "swap": 3,  # Swap
            "ch": 11,  # Controlled-H
            "ccx": 15,  # C3 gate: Toffoli
            "cswap": 17,  # Fredkin
            "crx": 5,  # Controlled RX rotation
            "cry": 4,  # Controlled RY rotation
            "crz": 4,  # Controlled RZ rotation
            "cu1": 5,  # Controlled phase rotation
            "cu3": 5,  # Controlled-U
            "rxx": 7,  # Two-qubit XX rotation
            "ryy": 7,
            "rzz": 3,  # Two-qubit ZZ rotation
            "rccx": 9,  # Relative-phase CCX
            "rc3x": 18,  # Relative-phase 3-controlled X gate
            "c3x": 27,  # 3-controlled X gate
            "c3sqrtx": 27,  # 3-controlled sqrt(X) gate
            "c4x": 87  # 4-controlled X gate

        }

        # OpenQASM native gate table, other gates are user-defined.
        self.GATE_TABLE = {**self.COMPOSITION_GATE_TABLE, **self.STANDARD_GATE_TABLE}

        # ==================================================================================
        # For the statistics of the number of CNOT or CX gate in the circuit

        # Number of CX in Standard gates
        self.STANDARD_CX_TABLE = {"r": 0,"u3": 0, "u2": 0, "u1": 0, "sx" : 0, "cx": 1, "id": 0, "x": 0, "y": 0, "z": 0, "h": 0,
                             "s": 0, "sdg": 0, "t": 0, "tdg": 0, "rx": 0, "ry": 0, "rz": 0, "c1": 0, "c2": 1}
        # Number of CX in Composition gates
        self.COMPOSITION_CX_TABLE = {"p":0,"cz": 1, "cy": 1, "swap": 3, "ch": 2, "ccx": 6, "cswap": 8, "crx": 2, "cry": 2,
                                "crz": 2, "cu1": 2, "cu3": 2, "rxx": 2, "rzz": 2, "ryy":2, "rccx": 3, "rc3x": 6, "c3x": 6,
                                "c3sqrtx": 6,
                                "c4x": 18}

        self.CX_TABLE = {**self.STANDARD_CX_TABLE, **self.COMPOSITION_CX_TABLE}

        self.USER_DEFINED_GATES = {}
        # Keywords in QASM that are currently not used
        self.other_keys = ["measure", "barrier", "OPENQASM", "include", "creg", "if", "reset"]
        self.measure_key = "measure"
        self.trigger_key = 'if'
        self.skip_keys = ["OPENQASM","include", "qreg", "creg","barrier", "reset","//"]

        self.circuit = qiskit.QuantumCircuit().from_qasm_str(qasm)
        self.circuit = RemoveBarriers()(self.circuit)

        self.preprocess_qasm()

    def preprocess_qasm(self):

        self.collate_gates()
        self.decompose_circuit()
        self.final_preprocessing()

    def get_op(self,line):
        """
        :param line: A line of QASM
        :return: The operation contained in the line of QASM
        """
        if line.find("(") != -1:
            line = line[:line.find("(")].strip()
        op = line.split(" ")[0].strip()
        return op
    
    def get_qubit_id(self,line):
        """
        Search for qubits that are active in a line of QASM code
        :param line: Line of QASM code
        :return: Qubits being used in the line of QASM code
        """
        line = line.strip(';')
        op_qubits = line.split(" ")[1].strip().split(',')
        qubit_ids = []
        for op_qubit in op_qubits:
            if '[' in op_qubit:
                qubit_prefix = op_qubit.split('[')[0]
                num = int(re.findall('^.*?\[[^\d]*(\d+)[^\d]*\].*$',op_qubit)[0])
                qubit_ids.append(qubit_prefix+str(num))
            else:
                qubit_ids = [x for x in self.qubit_labelled.keys() if op_qubit in x]
        return qubit_ids
    
    def collate_gates(self):
        """
        Preprocessing function for QASM String
        :return: None
        """
        gate_def = "gate"
        temporary_qasm = np.array([x.strip() for x in self.qasm.split('\n')])
        start_point,end_point = None,None
        to_remove = []
        for index,line in enumerate(temporary_qasm):
            line_contents = line.split(' ')
            if line_contents[0].strip() == gate_def:
                start_point = index
                gate_name = line_contents[1]
                qubit_count = len(line_contents[2].split(','))
            if line_contents[0].strip() == '}':
                end_point = index
            if start_point and end_point:
                gate_count = 0
                cx_count = 0
                for i in range(end_point-start_point-1):
                    print(temporary_qasm[start_point+i+1])
                    if '{' in temporary_qasm[start_point+i+1]:
                        continue
                    operation = self.get_op(temporary_qasm[start_point+i+1])
                    cx_count += self.CX_TABLE[operation]
                    gate_count += self.GATE_TABLE[operation]
                to_remove.append((start_point,end_point))
                end_point,start_point=None,None
                self.USER_DEFINED_GATES[gate_name] = None
                self.CX_TABLE[gate_name]=cx_count
                self.GATE_TABLE[gate_name]=gate_count
        valid_indexes = np.ones(len(temporary_qasm))
        for start,end in to_remove:
            valid_indexes[start:end+1] = 0
        temporary_qasm = temporary_qasm[valid_indexes.astype('bool')]
        self.qasm = temporary_qasm

    def decompose_circuit(self):
        gates = list(self.USER_DEFINED_GATES.keys())
        for _ in range(len(gates)):
            self.circuit = self.circuit.decompose(gates_to_decompose = gates)

    def final_preprocessing(self):
        """
        Extensive analysis of QASM string. Counts qubits, gets qubit ID's etc. This is the large preprocessing function
        :return:
        """
        qreg = "qreg"
        creg = "creg"
        regex_str = '^.*?\[[^\d]*(\d+)[^\d]*\].*$'
        regex_id_str = '(.*?)\s*\['
        # Break QASM into line by line commands
        qasm = self.qasm
        # Search for all qubit declaration lines
        qubit_count = [x for x in qasm if qreg in x]
        t_qubits = 0
        t_cbits = 0
        qbit_labelled = {}
        # Load all qubits into the qubit count and give them unique IDs
        for qubit_index in qubit_count:
            info_string = qubit_index.split(' ')[-1]
            qubit_id = re.findall(regex_id_str, info_string)[0]
            qbit_counts= int(re.findall(regex_str, qubit_index)[0])
            try:
                previous_cap = max(qbit_labelled.values()) +1
            except:
                previous_cap = 0
            for i in range(qbit_counts):
                qbit_labelled[str(qubit_id)+str(i)] = i + previous_cap
            t_qubits +=int(qbit_counts)
        # Search for all cbit declaration lines
        cbit_count = [x for x in qasm if creg in x]
        cbit_labelled = {}
        # Load all cbits into the cbit count and give them unique IDs
        for cbit_index in cbit_count:
            info_string = cbit_index.split(' ')[-1]
            cbit_id = re.findall(regex_id_str,info_string)[0]
            cbit_counts= int(re.findall(regex_str, cbit_index)[0])
            for i in range(cbit_counts):
                if len(cbit_labelled)==0:
                    cbit_labelled[str(cbit_id)+str(i)]=i
                else:
                    cbit_labelled[str(cbit_id)+str(i)]=i+max(cbit_labelled.values())
            t_cbits +=int(cbit_counts)
        # Remove all If statements from the QASM code, as we count these as gates.
        for i,line in enumerate(qasm):
            if self.trigger_key in line[:3]: # Fix this later - IF causes problems
                indx = line.find(' ')
                line = line[indx+1:]
                qasm[i] = line
        filtered_qasm = [x for x in qasm if not any(skip_key in x for skip_key in self.skip_keys)]
        filtered_qasm = [x for x in filtered_qasm if x != '']  # Drop any trailing end of lists such as ''
        measurement_count = len([x for x in filtered_qasm if self.measure_key in x])
        filtered_qasm = [x for x in filtered_qasm if not self.measure_key in x]
        self.measurement_count = measurement_count
        # Filter the QASM code for all lines containing strings within the "SKIP Keys" variable
        self.qubit_count = int(t_qubits)
        self.cbit_count = int(t_cbits)
        self.qubit_labelled = qbit_labelled
        self.cbit_labelled = cbit_labelled
        self.processed_qasm = filtered_qasm