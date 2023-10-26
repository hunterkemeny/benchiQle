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

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

# TODO list: 1. add remaining useful benchmarks from Red Queen (for solving the "gate" problem in red_queen, see Matthew's code)
#            2. Allow the user to choose which benchmarks and metrics to run on command line and as an import
#            3.  format the output of json files so that the title of the file mentions the compiler name that is being used (probably don't need the user to input this information, can get versioning in python)

#            4. Add support for other compilers (ptket, cirq, etc.)
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
        self.backend = backend
        self.num_runs = num_runs

        # TODO: make num_runs a dictionary element for each benchmark

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
                    self.full_benchmark_list.append({qasm_name: qiskit_circuit})
                    # TODO: improve the logic of this so that we iterate thru the metric_data dict to 
                    # populate these. And this same logic should be applied in run_benchmark
                    self.metric_data[qasm_name] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}

            elif benchmark == "medium":
                logger.info("medium_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in medium_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("medium_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append({qasm_name: qiskit_circuit})
                    self.metric_data[qasm_name] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
            
            elif benchmark == "large":
                logger.info("large_qasm: Converting from QASM into Qiskit circuits...")
                for qasm_name in large_qasm:
                    logger.info("Converting: " + qasm_name)
                    qasm = self.get_qasm_benchmark("large_qasm/" + qasm_name)
                    qiskit_circuit = QuantumCircuit.from_qasm_str(qasm)
                    self.full_benchmark_list.append({qasm_name: qiskit_circuit})
                    self.metric_data[qasm_name] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
            
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
                    self.metric_data["ft_circuit_1"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "ft_circuit_2":
                    self.full_benchmark_list.append({"ft_circuit_2": generate_ft_circuit_2("11111111")})
                    self.metric_data["ft_circuit_2"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "bv_mcm":
                    self.full_benchmark_list.append({"bv_mcm": build_bv_circuit("110011", True)})
                    self.metric_data["bv_mcm"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "bv":
                    self.full_benchmark_list.append({"bv": build_bv_circuit("110011")})
                    self.metric_data["bv"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "ipe":
                    self.full_benchmark_list.append({"ipe": quantum_phase_estimation(4, 1/8)})
                    self.metric_data["ipe"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "qpe":
                    self.full_benchmark_list.append({"qpe": build_ipe(4, 1/16)})
                    self.metric_data["qpe"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}
                elif benchmark == "EfficientSU2":
                    circuit = EfficientSU2(100, su2_gates=['rx'], entanglement='circular', reps=1)
                    params = circuit.parameters
                    # Does value of pi/4 make sense here?
                    values = [np.pi/4]*len(params)
                    binding = {param: value for param, value in zip(params, values)}
                    circuit = circuit.bind_parameters(binding)
                    self.full_benchmark_list.append({"EfficientSU2": circuit})
                    self.metric_data["EfficientSU2"] = {"speed (seconds)": [], "depth (gates)": [], "memory_footprint (MiB)": []}

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

        if "speed (seconds)" in self.metric_list:
            logger.info("Calculating speed...")
            # to get accurate time measurement, need to run transpilation without profiling
            start_time = time.time()
            transpiled_circuit = transpile(benchmark_circuit, backend=FakeWashingtonV2(), optimization_level=0)
            end_time = time.time()
            self.metric_data[benchmark_name]["speed (seconds)"].append(end_time - start_time)
        
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
                    ["depth (gates)", "speed (seconds)", "memory_footprint (MiB)"], 
                    {"compiler": "qiskit", "version": "10.2.0", "optimization_level": 0},
                    "qasm_simulator",
                    10)
    runner.run_benchmarks()



    