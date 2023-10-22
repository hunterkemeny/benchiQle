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
import gc
import tracemalloc
import multiprocessing

# Set up logging to display logs of level INFO and above
logging.basicConfig(level=logging.INFO)
# TODO: get speed metrics through logging in the future

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
        metric_data = {}
        metrics = Metrics()
        for benchmark in self.benchmark_list:
            self.run_benchmark(benchmark, metric_data, metrics)
        # TODO: perform aggregate statistics on metric_data and save to reports in JSON file

    @profile
    def transpile_in_process(self, benchmark, optimization_level, metric_data):
        # Force garbage collection
        gc.collect()
        start_time = time.time()
        transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=optimization_level) # TODO: add generality for compilers with compiler_dict
        end_time = time.time()
        metric_data['speed'] = end_time - start_time
        print(f"Time to transpile: {end_time - start_time}")
        # Force garbage collection again
        gc.collect()
        return transpiled_circuit
    
    def profile_func(self, benchmark, metric_data):

        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
    
        # To get accurate memory usage, need to multiprocess transpilation
        # Call your function
        
        with multiprocessing.Pool(1) as pool:
            circuit = pool.apply(self.transpile_in_process, (benchmark, self.compiler_dict["optimization_level"], metric_data))
            #mem_usage = memory_usage((self.transpile_in_process, (benchmark, self.compiler_dict["optimization_level"], metric_data)), interval=1, timeout=None)
        
        sys.stdout = old_stdout
        buffer_content = buffer.getvalue()

        # Do whatever you want with buffer_content
        print("Captured output:")
        print(buffer_content)

        # And you can also use the memory usage data stored in 'mem_usage'
        # print("Memory usage data:")
        # print(mem_usage)
        
        #return circuit

        
    def run_benchmark(self, benchmark, metric_data, metrics):
        """
        Run a single benchmark.

        :param benchmark_name: name of benchmark to be used 
        :param metric_data: dictionary containing all metric data
        """
        # TODO: Create functionality for providing user benchmark in other languages (e.g. cirq)
        
         
        
        # TODO: ask about if there is existing way to get speed from transpiler
        #transpiled_circuit = profile_transpilation()
        self.profile_func(benchmark, metric_data)
        metric_data['memory_footprint'] = ""

        # to get accurate time measurement, need to run transpilation without profiling
        start_time = time.time()
        transpiled_circuit = transpile(benchmark, backend=FakeWashingtonV2(), optimization_level=0)
        end_time = time.time()
        metric_data["speed"] = end_time - start_time
        
        qasm_string = transpiled_circuit.qasm()
        benchmark = Benchmark(qasm_string)

        if "depth" in self.metric_list:
            depth = metrics.get_circuit_depth(benchmark)
            metric_data["depth"] = depth
            #print(f"Depth of circuit: {depth}")
        

if __name__ == "__main__":
    runner = Runner(["ft_circuit_1", "ft_circuit_2", "bv_mcm", "bv", "ipe", "qpe"], 
                    ["depth"], 
                    {"compiler": "qiskit", "version": "10.2.0", "optimization_level": 0},
                    "qasm_simulator")
    runner.run_benchmarks()


    