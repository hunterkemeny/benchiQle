# TODO: add __init__.py to all folders
import sys
import io
import statistics

from benchmarks.benchmark import Benchmark
from benchmarks.benchmark import small_qasm
from benchmarks.benchmark import medium_qasm
from benchmarks.benchmark import large_qasm
# For benchmarks that are created with qiskit, import them here
from benchmarks.red_queen.run_ft import generate_ft_circuit_1, generate_ft_circuit_2
from benchmarks.red_queen.run_bv import build_bv_circuit
from benchmarks.red_queen.run_ipe import build_ipe
from benchmarks.red_queen.run_qpe import quantum_phase_estimation
from metrics.metrics import Metrics
from qiskit import *
from qiskit.circuit import Parameter
# TODO: should this be V2 or just FakeWashington
from qiskit.providers.fake_provider import FakeWashingtonV2
from qiskit.circuit.library import *
import json
import time
from memory_profiler import profile
from contextlib import redirect_stdout
import multiprocessing
import logging
import numpy as np

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.architecture import Architecture
from pytket.circuit import OpType, Node
from pytket.passes import *
from pytket.placement import NoiseAwarePlacement

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

# TODO list: 1. add remaining useful benchmarks from Red Queen (for solving the "gate" problem in red_queen, see Matthew's code)
#            2. Allow the user to choose which benchmarks and metrics to run on command line and as an import
#            3.  format the output of json files so that the title of the file mentions the compiler name that is being used (probably don't need the user to input this information, can get versioning in python)

#            4. *Add support for other compilers (ptket, cirq, etc.)
#            5.* Look at Luciano's message from a few weeks ago and run with those params (take a look at routing difficulty)
#            8.* Add examples
#            9.* Add graphs from luciano's file

#            6. Improve on logger output
#            7. Clean up code, add comments, go thru remainder of todos

class Runner:
    # TODO: Currently can only choose one transpiler option from only qiskit (no comparison); also add different versions
    # TODO: add functionality for running ptket transpilation and also cirq (do this after qiskit version comparisons, aggregate metrics, and adding more benchmarks)
    
    # TODO: metric_data should be structured as follows:
    #       top: benchmark name --> compiler version --> run number/aggregate --> metric name --> metric value
    def __init__(self, provided_benchmarks: list, metric_list: list, compiler_dict: dict, backend: str, num_runs: int):
        """
        :provided_benchmarks: list of benchmarks to be used --> [benchmark_name]
        :param metrics: list of metrics to be used --> [metric_name]
        :param compiler_dict: dictionary of compiler info --> {"compiler": "COMPILER_NAME", "version": "VERSION NUM", "optimization_level": OPTIMIZATION_LEVEL}
        :param backend: name of backend to be used --> "BACKEND_NAME"
        :param num_runs: number of times to run each benchmark
        """
        # TODO: add support for qasm strings that have the "gate" keyword
        # TODO: rename these
        self.ALLOWED_QISKIT_BENCHMARKS = [
            "ft_circuit_1",
            "ft_circuit_2",
            "bv_mcm",
            "bv",
            "ipe",
            "qpe",
            "EfficientSU2"
        ]
        
        self.provided_benchmarks = provided_benchmarks
        self.metric_list = metric_list
        self.compiler_dict = compiler_dict
        self.backend = FakeWashingtonV2()
        self.num_runs = num_runs

        # TODO: make num_runs a dictionary element for each benchmark

        self.full_benchmark_list = []
        self.metric_data = {}

        if compiler_dict["compiler"] == "tket":
            self.tket_pm = self.initialize_tket_pass_manager()

        self.preprocess_benchmarks()

    def initialize_tket_pass_manager(self):
        """
        Initialize a pass manager for tket.
        """
        # Build equivalent of tket backend, it can't represent heterogenous gate sets
        gateset = {OpType.X, OpType.SX, OpType.Rz, OpType.Measure, OpType.ECR, OpType.CZ}
        arch = Architecture(self.backend.coupling_map.graph.edge_list())
        averaged_node_gate_errors = {}
        averaged_edge_gate_errors = {}
        averaged_readout_errors = {Node(x[0]): self.backend.target["measure"][x].error for x in self.backend.target["measure"]}
        for qarg in self.backend.target.qargs:
            ops = [x for x in self.backend.target.operation_names_for_qargs(qarg) if x not in {"if_else", "measure", "delay"}]
            avg = 0#statistics.mean(self.backend.target[op][qarg].error for op in ops)
            if len(qarg) == 1:
                averaged_node_gate_errors[Node(qarg[0])] = avg
            else:
                averaged_edge_gate_errors[tuple(Node(x) for x in qarg)] = avg
        # BUild tket compilation sequence:
        passlist = [DecomposeBoxes()]
        passlist.append(FullPeepholeOptimise())
        mid_measure = True
        noise_aware_placement = NoiseAwarePlacement(
            arch,
            averaged_node_gate_errors,
            averaged_edge_gate_errors,
            averaged_readout_errors,
        )
        passlist.append(
            CXMappingPass(
                arch,
                noise_aware_placement,
                directed_cx=True,
                delay_measures=(not mid_measure),
            )
        )
        passlist.append(NaivePlacementPass(arch))
        passlist.extend(
            [
                KAKDecomposition(allow_swaps=False),
                CliffordSimp(False),
                SynthesiseTket(),
            ]
        )
        rebase_pass = auto_rebase_pass({OpType.X, OpType.SX, OpType.Rz, OpType.CZ})
        passlist.extend([rebase_pass, RemoveRedundancies()])
        passlist.append(
            SimplifyInitial(allow_classical=False, create_all_qubits=True)
        )
        tket_pm = SequencePass(passlist)
        return tket_pm

    def get_qasm_benchmark(self, qasm_name):
        with open("./benchmarks/" + f"{qasm_name}", "r") as f:
            qasm = f.read()
        return qasm
    
    def preprocess_benchmarks(self):
        """
        Preprocess benchmarks before running them. 
        """
        for benchmark in self.provided_benchmarks:
            # Should allow for both qasm string inputs and qiskit circuit inputs, 
            # but should just run benchmarks on the transpile/compile operations
            
            if benchmark == "small":
                logger.info("small_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in small_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("small_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append({qasm_name: qiskit_circuit})
                    # TODO: improve the logic of this so that we iterate thru the metric_data dict to 
                    # populate these. And this same logic should be applied in run_benchmark
                    self.metric_data[qasm_name] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}

            elif benchmark == "medium":
                logger.info("medium_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in medium_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("medium_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append({qasm_name: qiskit_circuit})
                    self.metric_data[qasm_name] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
            
            elif benchmark == "large":
                logger.info("large_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in large_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("large_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append({qasm_name: qiskit_circuit})
                    self.metric_data[qasm_name] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
            
            # TODO: add suport for red-queen qasm benchmarks
            # elif benchmark[-5:] == ".qasm":
            #     qasm = self.get_qasm_benchmark(benchmark)
            #     qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
            #     self.full_benchmark_list.append(qiskit_circuit)
            elif benchmark not in self.ALLOWED_QISKIT_BENCHMARKS:
                raise Exception(f"Invalid benchmark name: {benchmark}")
            else:
                # TODO: generalize inputs to these functions
                # TODO: Is there a way to make this more modular? Maybe just have sets of benchmarks here (e.g. small, red-queen) instead of setting each one
                #           could iterate over ALLOWED_QISKIT_BENCHMARKS and just check if benchmark is in that list
                if benchmark == "ft_circuit_1":
                    self.full_benchmark_list.append({"ft_circuit_1": generate_ft_circuit_1("11111111")})
                    self.metric_data["ft_circuit_1"] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "ft_circuit_2":
                    self.full_benchmark_list.append({"ft_circuit_2": generate_ft_circuit_2("11111111")})
                    self.metric_data["ft_circuit_2"] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "bv_mcm":
                    self.full_benchmark_list.append({"bv_mcm": build_bv_circuit("110011", True)})
                    self.metric_data["bv_mcm"] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "bv":
                    self.full_benchmark_list.append({"bv": build_bv_circuit("110011")})
                    self.metric_data["bv"] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "ipe":
                    self.full_benchmark_list.append({"ipe": quantum_phase_estimation(4, 1/8)})
                    self.metric_data["ipe"] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "qpe":
                    self.full_benchmark_list.append({"qpe": build_ipe(4, 1/16)})
                    self.metric_data["qpe"] = {"total time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "EfficientSU2":
                    self.metric_data["EfficientSU2"] = {"total_time (seconds)": [], "build_time (seconds)": [], "bind_time (seconds)": [], "transpile_time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": [], "version": self.compiler_dict["version"]}
                    start_time = time.perf_counter()
                    # qc = EfficientSU2(100, su2_gates=['rx', 'ry'], entanglement='circular', reps=1)
                    num_qubits = 5  # Number of qubits
                    reps = 1  # Number of repetitions of the SU2 layer

                    qc = QuantumCircuit(num_qubits)

                    # Parameters for rotations
                    theta = [Parameter(f'θ{i}') for i in range(num_qubits * 2 * reps)]

                    # Add EfficientSU2 layers
                    for rep in range(reps):
                        # Add RX and RY gates
                        for qubit in range(num_qubits):
                            qc.rx(theta[rep * num_qubits * 2 + qubit], qubit)
                            qc.ry(theta[rep * num_qubits * 2 + num_qubits + qubit], qubit)

                        # Add circular entanglement
                        for qubit in range(num_qubits):
                            qc.cx(qubit, (qubit + 1) % num_qubits)

                    # You can set the parameters to specific values if needed
                    param_values = {theta[i]: np.random.uniform(0, 2*np.pi) for i in range(len(theta))}
                    qc = qc.bind_parameters(param_values)

                    # Optionally, add measurements
                    qc.measure_all()
                    # num_qubits = 10
                    # qc = QuantumCircuit(num_qubits)
                    # qc.h(0)
                    # for i in range(num_qubits - 1):
                    #     qc.cx(0, i + 1)
                    #qc.measure_all()
                    build_done_time = time.perf_counter()
                    qc = qc.bind_parameters(np.random.rand(len(qc.parameters)))
                    bind_done_time = time.perf_counter()
                    self.metric_data["EfficientSU2"]["build_time (seconds)"].append(build_done_time - start_time)
                    self.metric_data["EfficientSU2"]["bind_time (seconds)"].append(bind_done_time - build_done_time)
                    self.full_benchmark_list.append({"EfficientSU2": qc})
                    

    def run_benchmarks(self):
        """
        Run all benchmarks in full_benchmark_list.
        """
        # TODO: perform aggregate statistics on metric_data with multiple runs (mean, median, range, variance)
        # TODO: figure out if these extra classes are necessary
        metrics = Metrics()
        logger_counter = 1
        for benchmark in self.full_benchmark_list:
            
            for run in range(self.num_runs):
                logger.info("Running benchmark " + str(logger_counter) + " of " + str(self.num_runs*len(self.full_benchmark_list)) + "...")
                self.run_benchmark(benchmark, metrics, run)
                logger_counter += 1
            
            self.postprocess_metrics(benchmark)

        with open('metrics.json', 'a') as json_file:
            json.dump(self.metric_data, json_file)
        # TODO postprocess metrics to include units and improve formatting

    @profile
    def transpile_in_process(self, benchmark, optimization_level):
        # TODO: build backend, import circuit, etc. within separate process
        #       because tket cannot be pickled.
        #       e.g. must build pass_list in separate process
        if self.compiler_dict["compiler"] == "tket":
            qc = qiskit_to_tk(benchmark)
            self.tket_pm.apply(qc)
            transpiled_circuit = tk_to_qiskit(qc)
        else:
            transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level) # TODO: add generality for compilers with compiler_dict
        return transpiled_circuit
    
    def profile_func(self, benchmark):
        # To get accurate memory usage, need to multiprocess transpilation
        with multiprocessing.Pool(1) as pool:
            circuit = pool.apply(self.transpile_in_process, (benchmark, self.compiler_dict["optimization_level"]))
        circuit = self.transpile_in_process(benchmark, self.compiler_dict["optimization_level"])
        return circuit
    
    def extract_memory_increments(self, filename, target_line):
        #increments = []
        # TODO: optimize this function
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Flag to check if the line with memory details is next
            for line in lines:
                if target_line in line:
                    parts = line.split()
                    if len(parts) > 3:  # Check to ensure the line has enough columns
                        increment_value = float(parts[3])  # The "Increment" value is in the 4th column
                        #increments.append(increment_value)
        return increment_value

        
    def run_benchmark(self, benchmark, metrics, run):
        """
        Run a single benchmark.

        :param benchmark_name: name of benchmark to be used 
        :param metric_data: dictionary containing all metric data
        """
        # TODO: Create functionality for providing user benchmark in other languages (e.g. cirq)

        # TODO Add progress status so the terminal isn't blank
        # TODO add variables to running the script so the user can decide benchmarks and metrics from the command line
        benchmark_name = list(benchmark.keys())[0]
        benchmark_circuit = list(benchmark.values())[0]
        # This line is resetting the dictionary. Should be adding to it instead

        # TODO: fix the logic here: the circuit will HAVE to be transpiled, 
        #       and there seems to be no reason NOT to collect time data. 
        #       So this should always be run at the top, and the option should
        #       be to DISPLAY (add to metric_data) rather than whether to RUN it.
        
        if "memory_footprint (MiB)" in self.metric_list:
            # Add memory_footprint to dictionary corresponding to this benchmark
            
            logger.info("Calculating memory footprint...")
            # Multiprocesss transpilation to get accurate memory usage
            self.profile_func(benchmark_circuit)
            # Replace this with the path to your file
            filename = 'memory.txt'
            # Replace this with the line you are targeting
            target_line = "transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level)"
            memory_data = self.extract_memory_increments(filename, target_line)
            # TODO: determine if units should be added here or in postprocessing
            self.metric_data[benchmark_name]["memory_footprint (MiB)"].append(memory_data)

        if "total_time (seconds)" in self.metric_list:
            logger.info("Calculating speed...")
            # to get accurate time measurement, need to run transpilation without profiling
            start_time = time.perf_counter()
            if self.compiler_dict["compiler"] == "tket":
                qc = qiskit_to_tk(benchmark_circuit)
                self.tket_pm.apply(qc)
                transpiled_circuit = tk_to_qiskit(qc)
            else:
                transpiled_circuit = transpile(benchmark_circuit, backend=FakeWashingtonV2(), optimization_level=0)
            end_time = time.perf_counter()
            self.metric_data[benchmark_name]["transpile_time (seconds)"].append(end_time - start_time)
            if benchmark_name == "EfficientSU2":
                self.metric_data[benchmark_name]["total_time (seconds)"].append(self.metric_data[benchmark_name]["transpile_time (seconds)"][-1] +  self.metric_data[benchmark_name]["transpile_time (seconds)"][-1] + self.metric_data[benchmark_name]["bind_time (seconds)"][-1] + self.metric_data[benchmark_name]["build_time (seconds)"][-1])
        
        if "depth (gates)" in self.metric_list:
            logger.info("Calculating depth...")
            qasm_string = transpiled_circuit.qasm()
            processed_qasm = Benchmark(qasm_string)
            depth = metrics.get_circuit_depth(processed_qasm)
            self.metric_data[benchmark_name]["depth (gates)"].append(depth)
    
    def postprocess_metrics(self, benchmark):
        """
        Postprocess metrics to include aggregate statistics.
        """
        # For each metric, calculate mean, median, range, variance, standard dev

        # aggregate:
        #   metric name --> aggregate statistics --> value
        # TODO: turn array into np array first, then do calculations
        benchmark_name = list(benchmark.keys())[0]
        self.metric_data[benchmark_name]["aggregate"] = {}
        for metric in self.metric_list:
            self.metric_data[benchmark_name]["aggregate"][metric] = {}

            self.metric_data[benchmark_name]["aggregate"][metric]["mean"] = np.mean(np.array(self.metric_data[benchmark_name][metric], dtype=float))
            self.metric_data[benchmark_name]["aggregate"][metric]["median"] = np.median(np.array(self.metric_data[benchmark_name][metric], dtype=float))
            self.metric_data[benchmark_name]["aggregate"][metric]["range"] = (np.min(np.array(self.metric_data[benchmark_name][metric], dtype=float)), np.max(np.array(self.metric_data[benchmark_name][metric], dtype=float)))
            self.metric_data[benchmark_name]["aggregate"][metric]["variance"] = np.var(np.array(self.metric_data[benchmark_name][metric], dtype=float))
            self.metric_data[benchmark_name]["aggregate"][metric]["standard_deviation"] = np.std(np.array(self.metric_data[benchmark_name][metric], dtype=float))              

        logger.info(self.metric_data)



if __name__ == "__main__":
    logger.debug("hello")
    runner = Runner(["EfficientSU2"], 
                    ["depth (gates)", "total_time (seconds)", "build_time (seconds)", "bind_time (seconds)", "transpile_time (seconds)", "memory_footprint (MiB)"], 
                    {"compiler": "tket", "version": 0, "optimization_level": 0}, # "version": str(sys.argv[1]),
                    "qasm_simulator",
                    1)
    runner.run_benchmarks()



    