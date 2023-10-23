class Metrics:

    # def __init__(self, metric):
    #     self.metric = metric


    # TODO: put all metrics as functions in Metric class
    #       load them to runner.py with metric_list. 

    """
    Metrics currently taken from QASMBench.
    Would likely be better to call QASMBench for these functions in the future.
    """

    def get_circuit_depth(self, benchmark):
        self.get_qubit_depths(benchmark)
        _, depth = self.get_maximum_qubit_depth(benchmark)
        return depth

    def get_qubit_depths(self, benchmark):
        """
        Get depth of a specific qubit
        :return:
        """
        qubit_depth = {}
        for gate in benchmark.processed_qasm:
            op = benchmark.get_op(gate)
            if op not in benchmark.GATE_TABLE:
                print(f"{op} not counted towards evaluation. Not a valid from default gate tables")
                continue
            else:
                qubit_id = benchmark.get_qubit_id(gate)
                for qubit in qubit_id:
                    if qubit not in qubit_depth.keys():
                        qubit_depth[qubit] = 0
                    qubit_depth[qubit] += 1
        self.qubit_depth = qubit_depth

    def get_maximum_qubit_depth(self, benchmark):
        """
        Get maximum qubit depth
        :return:
        """
        self.get_qubit_depths(benchmark)
        qubit_depths = self.qubit_depth
        max_value = max(qubit_depths.values())  # maximum value
        max_keys = [k for k, v in qubit_depths.items() if v == max_value][0]
        # getting all keys containing the `maximum`
        self.max_qubit_depth_id = max_keys
        self.max_qubit_depth = max_value
        return max_keys, max_value