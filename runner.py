# TODO: add __init__.py to all folders
from benchmarks.benchmark import Benchmark
from benchmarks.run_ft import generate_ft_circuit_1
from metrics.metrics import Metrics
from qiskit import *

class Runner:
    # TODO: Currently have tested one benchmark with one metric running with 
    #       the ability to choose transpiler option 
    # TODO: Should add param for backend
    def __init__(self, benchmark_list: list, metric_list: list, compiler_dict: dict,):
        """
        :benchmark_list: list of benchmarks to be used --> [benchmark_name]
        :param metrics: list of metrics to be used --> [metric_name]
        :param compiler_dict: dictionary of compiler info --> {"compiler": "COMPILER_NAME", "version": "VERSION NUM", "optimization_level": OPTIMIZATION_LEVEL}
        """
        self.benchmark_list = benchmark_list
        self.metric_list = metric_list
        self.compiler_dict = compiler_dict

    def run_benchmarks(self):
        """
        Run all benchmarks in benchmark_list.
        """
        for benchmark in self.benchmark_list:
            # TODO: Should collect metrics on the aggregate when running multiple benchmarks
            #       in addition to collecting metrics for individual benchmarks. This should influence
            #       the choice of data structure for metrics that is returned.
            self.run_benchmark(benchmark, self.metric_list, self.compiler_dict)
    
    def run_benchmark(self, benchmark_name):
        """
        Run a single benchmark.

        :param benchmark_name: name of benchmark to be used 
        """
        # TODO: Create functionality for providing user benchmark in form of qasm str 
        #       and also a circuit in other languages (e.g. cirq)
        
        # qiskit circ to qasm benchmark
        # TODO: currently hard-coding existing red-queen benchmarks. Next step is
        #       to impelement generality for all benchmarks using benchmark_list
        qiskit_circuit = generate_ft_circuit_1("11111111") # TODO: For qiskit circuits that are built with multiple options, we should create a different file for each one
        transpiled_circuit = transpile(qiskit_circuit, optimization_level=self.compiler_dict["optimization_level"]) # TODO: add generality for compilers with compiler_dict

        qasm_string = transpiled_circuit.qasm()
        benchmark = Benchmark(qasm_string)

        self.run_metrics(benchmark, self.metric_list, self.compiler_dict)

        # TODO: implement option to input qasm str, 
        #       transform into qiskit circ back, then back to qasm
    
    def get_qasm_benchmark(self, qasm_name):
        with open("./benchmarks/qasm/" + f"{qasm_name}", "r") as f:
            qasm = f.read()
        return qasm
    
    def run_metrics(self, benchmark):
        metrics = Metrics()
        # TODO: would be better to simply have for loop over all metrics in metric_list
        #       that retrieves metrics. 
        # TODO: decide on data structure for exporting metrics that is most helpful for visualization and interpretation
        if "depth" in self.metric_list:
            depth = metrics.get_circuit_depth(benchmark)
            print(f"Depth of circuit: {depth}")

if __name__ == "__main__":
    runner = Runner(["ft_circuit_1"], ["depth"], {"compiler": "qiskit", "version": "10.2.0", "optimization_level": 0})
    runner.run_benchmarks()


    