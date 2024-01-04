# TODO: add __init__.py to all folders
import sys
import statistics
import os

# TODO: possibly change this import
from benchmarking.benchmark import Benchmark

from metrics.metrics import Metrics
from qiskit import *
from qiskit.providers.fake_provider import FakeWashingtonV2
from qiskit.circuit.library import *
import json
import time
from memory_profiler import profile
import multiprocessing
import logging
import numpy as np

# PyTket imports
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.architecture import Architecture
from pytket.circuit import OpType, Node
from pytket.passes import *
from pytket.placement import NoiseAwarePlacement
from pytket.qasm import circuit_from_qasm

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

class Runner:
    def __init__(self, compiler_dict: dict, backend, num_runs: int, exclude_list=[]):
        """
        :param compiler_dict: dictionary of compiler info --> {"compiler": "COMPILER_NAME", "version": "VERSION NUM", "optimization_level": OPTIMIZATION_LEVEL}
        :param backend: name of backend to be used --> "BACKEND_NAME"
        :param num_runs: number of times to run each benchmark
        :param exclude_list: list of metrics to exclude from the benchmarking process
        """
        
        self.compiler_dict = compiler_dict
        self.backend = backend
        self.num_runs = num_runs
        self.exclude_list = exclude_list

        self.full_benchmark_list = []
        self.metric_data = {"metadata: ": self.compiler_dict}

        self.preprocess_benchmarks()

    def initialize_tket_pass_manager(self):
        """
        Initialize a pass manager for tket.
        """
        # Build equivalent of tket backend, it can't represent heterogenous gate sets
        arch = Architecture(self.backend.coupling_map.graph.edge_list())
        averaged_node_gate_errors = {}
        averaged_edge_gate_errors = {}
        averaged_readout_errors = {Node(x[0]): self.backend.target["measure"][x].error for x in self.backend.target["measure"]}
        for qarg in self.backend.target.qargs:
            ops = [x for x in self.backend.target.operation_names_for_qargs(qarg) if x not in {"if_else", "measure", "delay"}]
            avg = statistics.mean(self.backend.target[op][qarg].error for op in ops)
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
        with open("./benchmarking/benchmarks/" + f"{qasm_name}", "r") as f:
            qasm = f.read()
        return qasm
    
    def list_files(self, directory):
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    def preprocess_benchmarks(self):
        """
        Preprocess benchmarks before running them. 
        """
        benchmarks = self.list_files('./benchmarking/benchmarks/')
        for benchmark in benchmarks:
            
            qasm = self.get_qasm_benchmark(benchmark)
            logger.info("Converting " + benchmark + " to high-level circuit...")

            if self.compiler_dict["compiler"] == "tket":
                # TODO: determine if this should be a string or a file path
                tket_circuit = circuit_from_qasm(qasm)
                self.full_benchmark_list.append({benchmark: tket_circuit})
            elif self.compiler_dict["compiler"] == "qiskit":
                qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                self.full_benchmark_list.append({benchmark: qiskit_circuit})
            self.metric_data[benchmark] = {"total_time (seconds)": [], "build_time (seconds)": [], "bind_time (seconds)": [], "transpile_time (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": [], "version": self.compiler_dict["version"]}
            
    def run_benchmarks(self):
        """
        Run all benchmarks in full_benchmark_list.
        """
        # TODO: figure out if these extra classes are necessary
        # TODO: can likely remove metrics folder and class since matthew had another way of retrieving depth (make sure it is from qasm file)
        metrics = Metrics()
        logger_counter = 1
        for benchmark in self.full_benchmark_list:
            
            for _ in range(self.num_runs):
                logger.info("Running benchmark " + str(logger_counter) + " of " + str(self.num_runs*len(self.full_benchmark_list)) + "...")
                self.run_benchmark(benchmark, metrics)
                logger_counter += 1
            
            self.postprocess_metrics(benchmark)

        with open('metrics.json', 'a') as json_file:
            json.dump(self.metric_data, json_file)

    @profile
    def transpile_in_process(self, benchmark, optimization_level):
        if self.compiler_dict["compiler"] == "tket":
            tket_pm = self.initialize_tket_pass_manager()
            qc = qiskit_to_tk(benchmark)
            tket_pm.apply(qc)
            transpiled_circuit = tk_to_qiskit(qc)
        else:
            # TODO: Determine why we cannot use the backend from the constructor here (self.backend throwing error)
            transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level) # TODO: add generality for compilers with compiler_dict
        return transpiled_circuit
    
    def profile_func(self, benchmark):
        # To get accurate memory usage, need to multiprocess transpilation
        with multiprocessing.Pool(1) as pool:
            circuit = pool.apply(self.transpile_in_process, (benchmark, self.compiler_dict["optimization_level"]))
        return circuit
    
    def extract_memory_increments(self, filename, target_line):
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Flag to check if the line with memory details is next
            for line in lines:
                if target_line in line:
                    parts = line.split()
                    if len(parts) > 3:  # Check to ensure the line has enough columns
                        increment_value = float(parts[3])  # The "Increment" value is in the 4th column
        return increment_value

        
    def run_benchmark(self, benchmark, metrics):
        """
        Run a single benchmark.

        :param benchmark_name: name of benchmark to be used 
        :param metric_data: dictionary containing all metric data
        """

        # TODO add variables to running the script so the user can decide benchmarks and metrics from the command line
        benchmark_name = list(benchmark.keys())[0]
        benchmark_circuit = list(benchmark.values())[0]
        
        if "memory_footprint (MiB)" not in self.exclude_list:
            # Add memory_footprint to dictionary corresponding to this benchmark
            
            logger.info("Calculating memory footprint...")
            # Multiprocesss transpilation to get accurate memory usage
            self.profile_func(benchmark_circuit)
            filename = f'memory_{str(sys.argv[1])}_{str(sys.argv[2])}.txt'
            if self.compiler_dict["compiler"] == "tket":
                target_line = "tket_pm.apply(qc)"
            else:
                target_line = "transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level)"
            memory_data = self.extract_memory_increments(filename, target_line)
            self.metric_data[benchmark_name]["memory_footprint (MiB)"].append(memory_data)

        if "total_time (seconds)" not in self.exclude_list:
            logger.info("Calculating speed...")
            # to get accurate time measurement, need to run transpilation without profiling
            start_time = time.perf_counter()
            if self.compiler_dict["compiler"] == "tket":
                qc = qiskit_to_tk(benchmark_circuit)
                self.tket_pm.apply(qc)
                transpiled_circuit = tk_to_qiskit(qc)
            else:
                transpiled_circuit = transpile(benchmark_circuit, backend=self.backend, optimization_level=0)
            end_time = time.perf_counter()
            self.metric_data[benchmark_name]["transpile_time (seconds)"].append(end_time - start_time)
            if benchmark_name == "EfficientSU2":
                self.metric_data[benchmark_name]["total_time (seconds)"].append(self.metric_data[benchmark_name]["transpile_time (seconds)"][-1] + self.metric_data[benchmark_name]["bind_time (seconds)"][-1] + self.metric_data[benchmark_name]["build_time (seconds)"][-1])
        
        if "depth (gates)" not in self.exclude_list:
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
        benchmark_name = list(benchmark.keys())[0]
        self.metric_data[benchmark_name]["aggregate"] = {}
        # TODO: determine another way of retrieving what we previously had storded in matric_list
        for metric in self.metric_list:
            self.metric_data[benchmark_name]["aggregate"][metric] = {}
            self.metric_data[benchmark_name]["aggregate"][metric]["mean"] = np.mean(np.array(self.metric_data[benchmark_name][metric], dtype=float))
            self.metric_data[benchmark_name]["aggregate"][metric]["median"] = np.median(np.array(self.metric_data[benchmark_name][metric], dtype=float))
            self.metric_data[benchmark_name]["aggregate"][metric]["range"] = (np.min(np.array(self.metric_data[benchmark_name][metric], dtype=float)), np.max(np.array(self.metric_data[benchmark_name][metric], dtype=float)))
            self.metric_data[benchmark_name]["aggregate"][metric]["variance"] = np.var(np.array(self.metric_data[benchmark_name][metric], dtype=float))
            self.metric_data[benchmark_name]["aggregate"][metric]["standard_deviation"] = np.std(np.array(self.metric_data[benchmark_name][metric], dtype=float))              

        logger.info(self.metric_data)

if __name__ == "__main__":

    # TODO: change inputs so that there are multiple compilers, and that the inputs are from the command line
    runner = Runner({"compiler": str(sys.argv[1]), "version": str(sys.argv[2]), "optimization_level": 0},
                    # TODO: determine if we should pass in a coupling map (e.g. heavy hex) and then have an if statement
                    # that chooses the backend to map to. 
                    FakeWashingtonV2(),
                    2)
    runner.run_benchmarks()

    




    