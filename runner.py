# TODO: add __init__.py to all folders
import sys
import io

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
from qiskit.providers.fake_provider import FakeWashingtonV2
import json
import time
from memory_profiler import profile
from contextlib import redirect_stdout
import multiprocessing
import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

# TODO list: 1. add remaining useful benchmarks from Red Queen (for solving the "gate" problem in red_queen, see Matthew's code)
#            2. Add aggregate statistics and postprocessing
#            3. Add support for other versions of qiskit
#            4. compilers (ptket, cirq, etc.)
#            5. Clean up code, add comments, go thru remainder of todos
#            6. Add examples

class Runner:
    # TODO: Currently can only choose one transpiler option from only qiskit (no comparison); also add different versions
    # TODO: add functionality for running ptket transpilation and also cirq (do this after qiskit version comparisons, aggregate metrics, and adding more benchmarks)

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
            "qpe"
        ]
        
        self.provided_benchmarks = provided_benchmarks
        self.metric_list = metric_list
        self.compiler_dict = compiler_dict
        self.backend = backend
        self.num_runs = num_runs

        self.full_benchmark_list = []
        self.metric_data = {}

        self.preprocess_benchmarks()

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
                    self.full_benchmark_list.append(qiskit_circuit)

            elif benchmark == "medium":
                logger.info("medium_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in medium_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("medium_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append(qiskit_circuit)
            
            elif benchmark == "large":
                logger.info("large_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in large_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("large_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append(qiskit_circuit)
            
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
                if benchmark == "ft_circuit_1":
                    self.full_benchmark_list.append(generate_ft_circuit_1("11111111"))
                elif benchmark == "ft_circuit_2":
                    self.full_benchmark_list.append(generate_ft_circuit_2("11111111"))
                elif benchmark == "bv_mcm":
                    self.full_benchmark_list.append(build_bv_circuit("110011", True))
                elif benchmark == "bv":
                    self.full_benchmark_list.append(build_bv_circuit("110011"))
                elif benchmark == "ipe":
                    self.full_benchmark_list.append(quantum_phase_estimation(4, 1/8))
                elif benchmark == "qpe":
                    self.full_benchmark_list.append(build_ipe(4, 1/16))


    def run_benchmarks(self):
        """
        Run all benchmarks in full_benchmark_list.
        """
        # TODO: perform aggregate statistics on metric_data with multiple runs (mean, median, range, variance)
        # TODO: figure out if these extra classes are necessary
        metrics = Metrics()
        counter = 1
        for benchmark in self.full_benchmark_list:
            logger.info("Running benchmark " + str(counter) + " of " + str(len(self.full_benchmark_list)) + "...")
            self.run_benchmark(benchmark, metrics)
            counter += 1
        
        # TODO: Add functionality for red_queen and qasm_bench benchmarks (e.g. sabre)
        with open('metrics.json', 'w') as json_file:
            json.dump(self.metric_data, json_file)

        # TODO postprocess metrics to include units and improve formatting

    @profile
    def transpile_in_process(self, benchmark, optimization_level):
        transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level) # TODO: add generality for compilers with compiler_dict
        return transpiled_circuit
    
    def profile_func(self, benchmark):
        # To get accurate memory usage, need to multiprocess transpilation
        with multiprocessing.Pool(1) as pool:
            circuit = pool.apply(self.transpile_in_process, (benchmark, self.compiler_dict["optimization_level"]))
        return circuit
    
    def extract_memory_increments(self, filename, target_line):
        increments = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Flag to check if the line with memory details is next
            for line in lines:
                if target_line in line:
                    parts = line.split()
                    if len(parts) > 3:  # Check to ensure the line has enough columns
                        increment_value = float(parts[3])  # The "Increment" value is in the 4th column
                        increments.append(increment_value)
        return increments

        
    def run_benchmark(self, benchmark, metrics):
        """
        Run a single benchmark.

        :param benchmark_name: name of benchmark to be used 
        :param metric_data: dictionary containing all metric data
        """
        # TODO: Create functionality for providing user benchmark in other languages (e.g. cirq)

        # TODO Add progress status so the terminal isn't blank
        # TODO add variables to running the script so the user can decide benchmarks and metrics from the command line
        if "memory_footprint" in self.metric_list:
            logger.info("Calculating memory footprint...")
            # Multiprocesss transpilation to get accurate memory usage
            self.profile_func(benchmark)
            # Replace this with the path to your file
            filename = 'memory.txt'
            # Replace this with the line you are targeting
            target_line = "transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level)"
            memory_data = self.extract_memory_increments(filename, target_line)

            self.metric_data['memory_footprint'] = memory_data

        if "speed" in self.metric_list:
            logger.info("Calculating speed...")
            # to get accurate time measurement, need to run transpilation without profiling
            start_time = time.time()
            transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=0)
            end_time = time.time()
            self.metric_data["speed"] = end_time - start_time
        
        if "depth" in self.metric_list:
            logger.info("Calculating depth...")
            qasm_string = transpiled_circuit.qasm()
            benchmark = Benchmark(qasm_string)
            depth = metrics.get_circuit_depth(benchmark)
            self.metric_data["depth"] = depth
        

if __name__ == "__main__":
    logger.debug("hello")
    runner = Runner(["small"], 
                    ["depth", "speed", "memory_footprint"], 
                    {"compiler": "qiskit", "version": "10.2.0", "optimization_level": 0},
                    "qasm_simulator",
                    1)
    runner.run_benchmarks()



    