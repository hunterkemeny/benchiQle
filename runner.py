# TODO: add __init__.py to all folders
import sys
import io

from benchmarks.benchmark import Benchmark
# For benchmarks that are created with qiskit, import them here
from benchmarks.run_ft import generate_ft_circuit_1, generate_ft_circuit_2
from benchmarks.run_bv import build_bv_circuit
from benchmarks.run_ipe import build_ipe
from benchmarks.run_qpe import quantum_phase_estimation
from metrics.metrics import Metrics
from qiskit import *
from qiskit.providers.fake_provider import FakeWashingtonV2
import json
import logging
import time
from memory_profiler import profile
from memory_profiler import memory_usage
from memory_profiler import LineProfiler
from contextlib import redirect_stdout
import gc
import tracemalloc
import multiprocessing

class Runner:
    # TODO: Currently have tested one benchmark with one metric running with 
    #       the ability to choose transpiler option 
    def __init__(self, benchmark_list: list, metric_list: list, compiler_dict: dict, backend: str):
        """
        :benchmark_list: list of benchmarks to be used --> [benchmark_name]
        :param metrics: list of metrics to be used --> [metric_name]
        :param compiler_dict: dictionary of compiler info --> {"compiler": "COMPILER_NAME", "version": "VERSION NUM", "optimization_level": OPTIMIZATION_LEVEL}
        :param backend: name of backend to be used --> "BACKEND_NAME"
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
        
        self.benchmark_list = benchmark_list
        self.metric_list = metric_list
        self.compiler_dict = compiler_dict
        self.backend = backend

        self.metric_data = {}

        self.preprocess_benchmarks()

    def get_qasm_benchmark(self, qasm_name):
        with open("./benchmarks/qasm/" + f"{qasm_name}", "r") as f:
            qasm = f.read()
        return qasm
    
    def preprocess_benchmarks(self):
        """
        Preprocess benchmarks before running them. 
        """
        for benchmark in self.benchmark_list:
            if benchmark[-5:] == ".qasm":
                # Transform benchmark from qasm string to qiskit circuit
                # TODO: figure out if converting from qasm str to qiskit circuit and back to qasm with different
                #       transpiler options is useful.
                qasm = self.get_qasm_benchmark(benchmark)
                qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                self.benchmark_list[self.benchmark_list.index(benchmark)] = qiskit_circuit
            elif benchmark not in self.ALLOWED_QISKIT_BENCHMARKS:
                raise Exception(f"Invalid benchmark name: {benchmark}")
            else:
                # TODO: generalize inputs to these functions
                # TODO: Is there a way to make this more modular?
                if benchmark == "ft_circuit_1":
                    self.benchmark_list[self.benchmark_list.index(benchmark)] = generate_ft_circuit_1("11111111")
                elif benchmark == "ft_circuit_2":
                    self.benchmark_list[self.benchmark_list.index(benchmark)] = generate_ft_circuit_2("11111111")
                elif benchmark == "bv_mcm":
                    self.benchmark_list[self.benchmark_list.index(benchmark)] = build_bv_circuit("110011", True)
                elif benchmark == "bv":
                    self.benchmark_list[self.benchmark_list.index(benchmark)] = build_bv_circuit("110011")
                elif benchmark == "ipe":
                    self.benchmark_list[self.benchmark_list.index(benchmark)] = quantum_phase_estimation(4, 1/8)
                elif benchmark == "qpe":
                    self.benchmark_list[self.benchmark_list.index(benchmark)] = build_ipe(4, 1/16)

    def run_benchmarks(self):
        """
        Run all benchmarks in benchmark_list.
        """
        # TODO: implement JSON for storing metric information.
        #       should have object for each individual benchmark along with
        #       an aggregate object for all benchmarks.
        metrics = Metrics()
        for benchmark in self.benchmark_list:
            self.run_benchmark(benchmark, metrics)
        # TODO: perform aggregate statistics on metric_data and save to reports in JSON file

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

        # TODO Add progress status so the terminal isn't simply blank
        # TODO add variables to running the script so the user can decide benchmarks and metrics from the command line
        #       also add 
        if "memory_footprint" in self.metric_list:
            # Multiprocesss transpilation to get accurate memory usage
            self.profile_func(benchmark)
            # Replace this with the path to your file
            filename = 'memory.txt'
            # Replace this with the line you are targeting
            target_line = "transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level)"
            memory_data = self.extract_memory_increments(filename, target_line)

            self.metric_data['memory_footprint'] = memory_data

        if "speed" in self.metric_list:
            # to get accurate time measurement, need to run transpilation without profiling
            start_time = time.time()
            transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=0)
            end_time = time.time()
            self.metric_data["speed"] = end_time - start_time
        
        if "depth" in self.metric_list:
            qasm_string = transpiled_circuit.qasm()
            benchmark = Benchmark(qasm_string)
            depth = metrics.get_circuit_depth(benchmark)
            self.metric_data["depth"] = depth
        

if __name__ == "__main__":
    runner = Runner(["ft_circuit_1", "ft_circuit_2", "bv_mcm", "bv", "ipe", "qpe"], 
                    ["depth", "speed", "memory_footprint"], 
                    {"compiler": "qiskit", "version": "10.2.0", "optimization_level": 0},
                    "qasm_simulator")
    runner.run_benchmarks()
    print(runner.metric_data)



    