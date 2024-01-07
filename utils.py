# PyTket imports
from pytket.architecture import Architecture
from pytket.circuit import OpType, Node
from pytket.passes import *
from pytket.placement import NoiseAwarePlacement
from pytket.qasm import circuit_from_qasm
from pytket.qasm import circuit_to_qasm_str

import statistics

def initialize_tket_pass_manager(backend):
    """
    Initialize a pass manager for tket.
    """
    # Build equivalent of tket backend, it can't represent heterogenous gate sets
    arch = Architecture(backend.coupling_map.graph.edge_list())
    averaged_node_gate_errors = {}
    averaged_edge_gate_errors = {}
    averaged_readout_errors = {Node(x[0]): backend.target["measure"][x].error for x in backend.target["measure"]}
    for qarg in backend.target.qargs:
        ops = [x for x in backend.target.operation_names_for_qargs(qarg) if x not in {"if_else", "measure", "delay"}]
        errors = [backend.target[op][qarg].error for op in ops if backend.target[op][qarg].error is not None]
        if errors:
            avg = statistics.mean(errors)
        else:
            avg = 0  # or some other default value
        
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