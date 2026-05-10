
import numpy as np
import torch
import pennylane as qml
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import normalize as sklearn_normalize, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import time
import warnings
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy
import sys
import math


@dataclass
class ParallelQuantumConfig:
    split_strategy: str = 'fixed_size'
    num_parts: int = 12
    features_per_part: int = None
    total_processed_dim: int = 768

    # Quantum circuit settings
    encoding_type: str = 'amplitude'
    n_qubits: int = 6
    n_layers: int = 2
    ansatz_type: str = 'strongly_entangling'
    ansatz_params: Dict = field(default_factory=lambda: {})
    measurement_type: str = 'expval_z'
    measurement_params: Dict = field(default_factory=lambda: {})

    feature_map_reps: int = 1
    entanglement: str = 'linear'
    pauli_strings: List[str] = field(default_factory=lambda: ['Z', 'ZZ'])

    use_dim_reduction: bool = True
    dim_reduction_layers: List[int] = field(default_factory=lambda: [256, 64])
    input_dim: int = None

    device_name: str = 'default.qubit'
    diff_method: str = 'backprop'
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 80
    optimizer: str = 'adam'
    dropout_rate: float = 0.3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    aggregation: str = 'concat'

    def __post_init__(self):
        if self.features_per_part is None and self.num_parts > 0:
            self.features_per_part = self.total_processed_dim // self.num_parts
        elif self.num_parts is None and self.features_per_part is not None:
            self.num_parts = self.total_processed_dim // self.features_per_part

    @property
    def measurements_per_qubit(self) -> int:
        if self.measurement_type == 'multi_xyz':
            return 3
        elif self.measurement_type in ['multi_xz', 'multi_xy', 'multi_yz']:
            return 2
        elif self.measurement_type in ['expval_z', 'expval_x', 'expval_y']:
            return 1
        else:
            return 1
    
class ParallelEncodingStrategies:

    @staticmethod
    def amplitude_encode(features: torch.Tensor, n_qubits: int, part_idx: int = 0):
        batch_size = features.shape[0]
        n_features = features.shape[1]

        required_qubits = int(np.ceil(np.log2(n_features)))
        if required_qubits > n_qubits:
            raise ValueError(f"Part {part_idx}: Need at least {required_qubits} qubits for {n_features} features")

        target_dim = 2 ** n_qubits
        if n_features < target_dim:
            padding = torch.zeros(batch_size, target_dim - n_features, device=features.device)
            features_padded = torch.cat([features, padding], dim=1)
        else:
            features_padded = features[:, :target_dim]

        norms = torch.norm(features_padded, dim=1, keepdim=True)
        norms = norms + 1e-10
        normalized_features = features_padded / norms

        qml.AmplitudeEmbedding(features=normalized_features, wires=range(n_qubits), normalize=False)

    @staticmethod
    def angle_encode(features: torch.Tensor, n_qubits: int, rotation: str = 'RY', part_idx: int = 0):
        batch_size = features.shape[0]
        if features.shape[1] > n_qubits:
            features = features[:, :n_qubits]
        if features.shape[1] < n_qubits:
            padding = torch.zeros(batch_size, n_qubits - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)

        scaled_features = 2 * np.pi * torch.sigmoid(features)

        for i in range(n_qubits):
            if rotation == 'RY':
                qml.RY(scaled_features[:, i], wires=i)
            elif rotation == 'RX':
                qml.RX(scaled_features[:, i], wires=i)
            elif rotation == 'RZ':
                qml.RZ(scaled_features[:, i], wires=i)

    @staticmethod
    def phase_encode(features: torch.Tensor, n_qubits: int, part_idx: int = 0):
        batch_size = features.shape[0]
        if features.shape[1] > n_qubits:
            features = features[:, :n_qubits]
        if features.shape[1] < n_qubits:
            padding = torch.zeros(batch_size, n_qubits - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)

        scaled_features = 2 * np.pi * torch.sigmoid(features)

        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_qubits):
            qml.RZ(scaled_features[:, i], wires=i)

    @staticmethod
    def dense_encode(features: torch.Tensor, n_qubits: int, part_idx: int = 0):
        batch_size = features.shape[0]
        n_features = features.shape[1]

        if n_features > 2 * n_qubits:
            features = features[:, :2*n_qubits]

        n_pairs = min(n_qubits, (n_features + 1) // 2)

        for i in range(n_pairs):
            if 2*i + 1 < n_features:
                theta = 2 * np.pi * torch.sigmoid(features[:, 2*i])
                phi = 2 * np.pi * torch.sigmoid(features[:, 2*i + 1])
                qml.RY(theta, wires=i)
                qml.RZ(phi, wires=i)
            elif 2*i < n_features:
                theta = 2 * np.pi * torch.sigmoid(features[:, 2*i])
                qml.RY(theta, wires=i)

        for i in range(n_pairs, n_qubits):
            qml.Identity(wires=i)

    @staticmethod
    def iqp_encode(features: torch.Tensor, n_qubits: int, n_repeats: int = 2, part_idx: int = 0):
        batch_size = features.shape[0]

        if features.shape[1] > n_qubits:
            features = features[:, :n_qubits]
        if features.shape[1] < n_qubits:
            padding = torch.zeros(batch_size, n_qubits - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)

        for r in range(n_repeats):
            for i in range(n_qubits):
                scaled = 2 * np.pi * torch.sigmoid(features[:, i])
                qml.RZ(scaled, wires=i)
                qml.RX(scaled, wires=i)

            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[n_qubits - 1, 0])

    @staticmethod
    def efficient_su2_encode(features: torch.Tensor, n_qubits: int, reps: int = 1, part_idx: int = 0):

        batch_size = features.shape[0]
        n_features = features.shape[1]

        max_features = n_qubits * (2 * reps + 1)

        if n_features > max_features:
            features = features[:, :max_features]
            n_features = max_features


        params_per_layer = n_qubits * 2

        for layer in range(reps):
            start_idx = layer * params_per_layer
            for i in range(n_qubits):
                if start_idx + i < n_features:
                    scaled = 2 * np.pi * torch.sigmoid(features[:, start_idx + i])
                    qml.RY(scaled, wires=i)

            start_idx_rz = start_idx + n_qubits
            for i in range(n_qubits):
                if start_idx_rz + i < n_features:
                    scaled = 2 * np.pi * torch.sigmoid(features[:, start_idx_rz + i])
                    qml.RZ(scaled, wires=i)

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])

        if reps > 0:
            final_start = reps * params_per_layer
            for i in range(n_qubits):
                if final_start + i < n_features:
                    scaled = 2 * np.pi * torch.sigmoid(features[:, final_start + i])
                    qml.RY(scaled, wires=i)

    @staticmethod
    def z_feature_map_encode(features: torch.Tensor, n_qubits: int, reps: int = 1, part_idx: int = 0):

        batch_size = features.shape[0]

        if features.shape[1] > n_qubits:
            features = features[:, :n_qubits]
        if features.shape[1] < n_qubits:
            padding = torch.zeros(batch_size, n_qubits - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)

        scaled_features = 2 * np.pi * torch.sigmoid(features)

        for rep in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            for i in range(n_qubits):
                qml.RZ(scaled_features[:, i], wires=i)

    @staticmethod
    def zz_feature_map_encode(features: torch.Tensor, n_qubits: int, reps: int = 1, entanglement: str = 'linear', part_idx: int = 0):

        batch_size = features.shape[0]

        if features.shape[1] > n_qubits:
            features = features[:, :n_qubits]
        if features.shape[1] < n_qubits:
            padding = torch.zeros(batch_size, n_qubits - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)

        scaled_features = 2 * np.pi * torch.sigmoid(features)

        for rep in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            for i in range(n_qubits):
                qml.RZ(scaled_features[:, i], wires=i)

            if entanglement == 'linear':
                for i in range(n_qubits - 1):
                    phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, i+1])
                    qml.IsingZZ(phi, wires=[i, i+1])
            elif entanglement == 'circular':
                for i in range(n_qubits - 1):
                    phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, i+1])
                    qml.IsingZZ(phi, wires=[i, i+1])
                phi_circ = 2 * (np.pi - scaled_features[:, n_qubits-1]) * (np.pi - scaled_features[:, 0])
                qml.IsingZZ(phi_circ, wires=[n_qubits-1, 0])
            elif entanglement == 'full':
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, j])
                        qml.IsingZZ(phi, wires=[i, j])

    @staticmethod
    def pauli_feature_map_encode(features: torch.Tensor, n_qubits: int, reps: int = 1,entanglement: str = 'linear', pauli_strings: List[str] = None, part_idx: int = 0):

        batch_size = features.shape[0]

        if pauli_strings is None:
            pauli_strings = ['Z', 'ZZ']

        if features.shape[1] > n_qubits:
            features = features[:, :n_qubits]
        if features.shape[1] < n_qubits:
            padding = torch.zeros(batch_size, n_qubits - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)

        scaled_features = 2 * np.pi * torch.sigmoid(features)

        for rep in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            for pauli in pauli_strings:
                if pauli == 'Z':
                    for i in range(n_qubits):
                        qml.RZ(scaled_features[:, i], wires=i)

                elif pauli == 'X':
                    for i in range(n_qubits):
                        qml.RX(scaled_features[:, i], wires=i)

                elif pauli == 'Y':
                    for i in range(n_qubits):
                        qml.RY(scaled_features[:, i], wires=i)

                elif pauli == 'ZZ':
                    if entanglement == 'linear':
                        for i in range(n_qubits - 1):
                            phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, i+1])
                            qml.IsingZZ(phi, wires=[i, i+1])
                    elif entanglement == 'circular':
                        for i in range(n_qubits - 1):
                            phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, i+1])
                            qml.IsingZZ(phi, wires=[i, i+1])
                        phi_circ = 2 * (np.pi - scaled_features[:, n_qubits-1]) * (np.pi - scaled_features[:, 0])
                        qml.IsingZZ(phi_circ, wires=[n_qubits-1, 0])

                elif pauli == 'XX':
                    if entanglement == 'linear':
                        for i in range(n_qubits - 1):
                            phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, i+1])
                            qml.IsingXX(phi, wires=[i, i+1])

                elif pauli == 'YY':
                    if entanglement == 'linear':
                        for i in range(n_qubits - 1):
                            phi = 2 * (np.pi - scaled_features[:, i]) * (np.pi - scaled_features[:, i+1])
                            qml.IsingYY(phi, wires=[i, i+1])
class ParallelAnsatzStrategies:

    @staticmethod
    def strongly_entangling_ansatz(weights: torch.Tensor, wires: List[int], n_layers: int, n_qubits: int):
        qml.StronglyEntanglingLayers(weights, wires=wires)

    @staticmethod
    def basic_entangler_ansatz(weights: torch.Tensor, wires: List[int], n_layers: int, n_qubits: int):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=wires[i])
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
            if n_qubits > 2:
                qml.CNOT(wires=[wires[n_qubits-1], wires[0]])

    @staticmethod
    def hardware_efficient_ansatz(weights: torch.Tensor, wires: List[int], n_layers: int, n_qubits: int):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=wires[i])
                qml.RY(weights[layer, i, 1], wires=wires[i])
                qml.RZ(weights[layer, i, 2], wires=wires[i])

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.CNOT(wires=[wires[n_qubits-1], wires[0]])

    @staticmethod
    def simplified_two_design_ansatz(weights: torch.Tensor, wires: List[int], n_layers: int, n_qubits: int):
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=wires[i])
                qml.RY(weights[layer, i, 1], wires=wires[i])
                qml.RZ(weights[layer, i, 2], wires=wires[i])

            for i in range(0, n_qubits-1, 2):
                qml.CNOT(wires=[wires[i], wires[i+1]])
            for i in range(1, n_qubits-1, 2):
                qml.CNOT(wires=[wires[i], wires[i+1]])

    @staticmethod
    def apply_ansatz(weights: torch.Tensor, wires: List[int], config: ParallelQuantumConfig):
        n_layers = config.n_layers
        n_qubits = config.n_qubits

        if config.ansatz_type == 'strongly_entangling':
            ParallelAnsatzStrategies.strongly_entangling_ansatz(weights, wires, n_layers, n_qubits)
        elif config.ansatz_type == 'basic_entangler':
            ParallelAnsatzStrategies.basic_entangler_ansatz(weights, wires, n_layers, n_qubits)
        elif config.ansatz_type == 'hardware_efficient':
            ParallelAnsatzStrategies.hardware_efficient_ansatz(weights, wires, n_layers, n_qubits)
        elif config.ansatz_type == 'simplified_two_design':
            ParallelAnsatzStrategies.simplified_two_design_ansatz(weights, wires, n_layers, n_qubits)
class ParallelMeasurementStrategies:

    @staticmethod
    def expval_z_measurement(n_qubits: int):
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    @staticmethod
    def expval_x_measurement(n_qubits: int):
        return [qml.expval(qml.PauliX(w)) for w in range(n_qubits)]

    @staticmethod
    def expval_y_measurement(n_qubits: int):
        return [qml.expval(qml.PauliY(w)) for w in range(n_qubits)]

    @staticmethod
    def multi_xz_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            measurements.append(qml.expval(qml.PauliX(w)))
            measurements.append(qml.expval(qml.PauliZ(w)))
        return measurements

    @staticmethod
    def multi_xy_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            measurements.append(qml.expval(qml.PauliX(w)))
            measurements.append(qml.expval(qml.PauliY(w)))
        return measurements

    @staticmethod
    def multi_yz_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            measurements.append(qml.expval(qml.PauliY(w)))
            measurements.append(qml.expval(qml.PauliZ(w)))
        return measurements

    @staticmethod
    def multi_xyz_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            measurements.append(qml.expval(qml.PauliX(w)))
            measurements.append(qml.expval(qml.PauliY(w)))
            measurements.append(qml.expval(qml.PauliZ(w)))
        return measurements

    @staticmethod
    def expval_zx_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            if w % 2 == 0:
                measurements.append(qml.expval(qml.PauliZ(w)))
            else:
                measurements.append(qml.expval(qml.PauliX(w)))
        return measurements

    @staticmethod
    def expval_zy_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            if w % 2 == 0:
                measurements.append(qml.expval(qml.PauliZ(w)))
            else:
                measurements.append(qml.expval(qml.PauliY(w)))
        return measurements

    @staticmethod
    def expval_zxy_measurement(n_qubits: int):
        measurements = []
        for w in range(n_qubits):
            if w % 3 == 0:
                measurements.append(qml.expval(qml.PauliZ(w)))
            elif w % 3 == 1:
                measurements.append(qml.expval(qml.PauliX(w)))
            else:
                measurements.append(qml.expval(qml.PauliY(w)))
        return measurements

    @staticmethod
    def correlated_measurement(n_qubits: int, n_pairs: int = None):
        if n_pairs is None:
            n_pairs = n_qubits // 2

        measurements = []
        for i in range(n_pairs):
            if 2*i + 1 < n_qubits:
                measurements.append(qml.expval(qml.PauliZ(2*i) @ qml.PauliZ(2*i+1)))
        return measurements

    @staticmethod
    def trainable_measurement(n_qubits: int, params: torch.Tensor = None):
        if params is not None:
            for w in range(n_qubits):
                qml.RY(params[w], wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    @staticmethod
    def measure(measurement_type: str, n_qubits: int, params: Dict = None):
        if measurement_type == 'expval_z':
            return ParallelMeasurementStrategies.expval_z_measurement(n_qubits)
        elif measurement_type == 'expval_x':
            return ParallelMeasurementStrategies.expval_x_measurement(n_qubits)
        elif measurement_type == 'expval_y':
            return ParallelMeasurementStrategies.expval_y_measurement(n_qubits)
        elif measurement_type == 'multi_xz':
            return ParallelMeasurementStrategies.multi_xz_measurement(n_qubits)
        elif measurement_type == 'multi_xy':
            return ParallelMeasurementStrategies.multi_xy_measurement(n_qubits)
        elif measurement_type == 'multi_yz':
            return ParallelMeasurementStrategies.multi_yz_measurement(n_qubits)
        elif measurement_type == 'multi_xyz':
            return ParallelMeasurementStrategies.multi_xyz_measurement(n_qubits)
        elif measurement_type == 'expval_zx':
            return ParallelMeasurementStrategies.expval_zx_measurement(n_qubits)
        elif measurement_type == 'expval_zy':
            return ParallelMeasurementStrategies.expval_zy_measurement(n_qubits)
        elif measurement_type == 'expval_zxy':
            return ParallelMeasurementStrategies.expval_zxy_measurement(n_qubits)
        elif measurement_type == 'correlated':
            n_pairs = params.get('n_pairs', None) if params else None
            return ParallelMeasurementStrategies.correlated_measurement(n_qubits, n_pairs)
        elif measurement_type == 'trainable':
            trainable_params = params.get('trainable_params', None) if params else None
            return ParallelMeasurementStrategies.trainable_measurement(n_qubits, trainable_params)
        else:
            return ParallelMeasurementStrategies.expval_z_measurement(n_qubits)
def create_parallel_quantum_circuit(part_idx: int, config: ParallelQuantumConfig):
    n_qubits = config.n_qubits
    dev = qml.device(config.device_name, wires=n_qubits, shots=None)

    @qml.qnode(dev, interface="torch", diff_method=config.diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        if config.encoding_type == 'amplitude':
            ParallelEncodingStrategies.amplitude_encode(inputs, n_qubits, part_idx)
        elif config.encoding_type == 'angle':
            rotation = config.ansatz_params.get('rotation', 'RY')
            ParallelEncodingStrategies.angle_encode(inputs, n_qubits, rotation, part_idx)
        elif config.encoding_type == 'phase':
            ParallelEncodingStrategies.phase_encode(inputs, n_qubits, part_idx)
        elif config.encoding_type == 'dense':
            ParallelEncodingStrategies.dense_encode(inputs, n_qubits, part_idx)
        elif config.encoding_type == 'iqp':
            n_repeats = config.ansatz_params.get('iqp_repeats', 2)
            ParallelEncodingStrategies.iqp_encode(inputs, n_qubits, n_repeats, part_idx)
        elif config.encoding_type == 'efficient_su2':
            reps = config.feature_map_reps
            ParallelEncodingStrategies.efficient_su2_encode(inputs, n_qubits, reps, part_idx)
        elif config.encoding_type == 'z_feature_map':
            reps = config.feature_map_reps
            ParallelEncodingStrategies.z_feature_map_encode(inputs, n_qubits, reps, part_idx)
        elif config.encoding_type == 'zz_feature_map':
            reps = config.feature_map_reps
            entanglement = config.entanglement
            ParallelEncodingStrategies.zz_feature_map_encode(inputs, n_qubits, reps, entanglement, part_idx)
        elif config.encoding_type == 'pauli_feature_map':
            reps = config.feature_map_reps
            entanglement = config.entanglement
            pauli_strings = config.pauli_strings
            ParallelEncodingStrategies.pauli_feature_map_encode(inputs, n_qubits, reps, entanglement, pauli_strings, part_idx)
        else:
            ParallelEncodingStrategies.amplitude_encode(inputs, n_qubits, part_idx)

        if config.encoding_type not in ['efficient_su2', 'z_feature_map', 'zz_feature_map', 'pauli_feature_map']:
            ParallelAnsatzStrategies.apply_ansatz(weights, range(n_qubits), config)
        else:
            ParallelAnsatzStrategies.apply_ansatz(weights, range(n_qubits), config)

        return ParallelMeasurementStrategies.measure(config.measurement_type, n_qubits, config.measurement_params)

    return circuit

class ParallelQuantumLayer(nn.Module):

    def __init__(self, input_dim: int, config: ParallelQuantumConfig):
        super().__init__()
        self.config = config
        self.num_parts = config.num_parts
        self.features_per_part = config.features_per_part

        self.quantum_circuits = nn.ModuleList()
        self.qlayers = nn.ModuleList()

        for i in range(self.num_parts):
            circuit = create_parallel_quantum_circuit(i, config)
            weight_shape = (config.n_layers, config.n_qubits, 3)
            qlayer = qml.qnn.TorchLayer(circuit, {"weights": weight_shape})
            self.qlayers.append(qlayer)

        print(f"\n Parallel Quantum Layer")
        print(f"Number of parallel circuits: {self.num_parts}")
        print(f"Features per part: {self.features_per_part}")
        print(f"Total processed features: {self.num_parts * self.features_per_part}")
        print(f"Encoding: {config.encoding_type}")
        print(f"Ansatz: {config.ansatz_type}")
        print(f"Qubits per circuit: {config.n_qubits}")
        print(f"Measurements per qubit: {config.measurements_per_qubit if hasattr(config, 'measurements_per_qubit') else 1}")
        print(f"Output dimension per circuit: {config.n_qubits * (config.measurements_per_qubit if hasattr(config, 'measurements_per_qubit') else 1)}")
        print(f"Layers per circuit: {config.n_layers}")

    def forward(self, x):
        batch_size = x.shape[0]

        total_needed = self.num_parts * self.features_per_part
        if x.shape[1] < total_needed:
            padding = torch.zeros(batch_size, total_needed - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)

        parts = torch.split(x, self.features_per_part, dim=1)

        outputs = []
        for i, qlayer in enumerate(self.qlayers):
            if i < len(parts):
                part_output = qlayer(parts[i])
                outputs.append(part_output)

        stacked = torch.stack(outputs, dim=0)

        if self.config.aggregation == 'concat':
            result = stacked.permute(1, 0, 2).reshape(batch_size, -1)
        elif self.config.aggregation == 'mean':
            result = stacked.mean(dim=0)
        elif self.config.aggregation == 'sum':
            result = stacked.sum(dim=0)
        else:
            result = stacked.permute(1, 0, 2).reshape(batch_size, -1)

        return result

class ParallelHybridQuantumClassifier(nn.Module):

    def __init__(self, input_dim: int, config: ParallelQuantumConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        config.input_dim = input_dim

        if config.use_dim_reduction:
            layers = []
            prev_dim = input_dim
            for hidden_dim in config.dim_reduction_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, config.total_processed_dim))
            self.dim_reduction = nn.Sequential(*layers)
            processed_dim = config.total_processed_dim
        else:
            self.dim_reduction = nn.Identity()
            processed_dim = input_dim

        self.parallel_quantum = ParallelQuantumLayer(processed_dim, config)

        if config.measurement_type == 'multi_xyz':
            measurement_dim = config.n_qubits * 3
        elif config.measurement_type in ['multi_xz', 'multi_xy', 'multi_yz']:
            measurement_dim = config.n_qubits * 2
        elif config.measurement_type in ['expval_z', 'expval_x', 'expval_y']:
            measurement_dim = config.n_qubits
        else:
            measurement_dim = config.n_qubits

        if config.aggregation == 'concat':
            quantum_output_dim = config.num_parts * measurement_dim
        else:
            quantum_output_dim = measurement_dim

        post_layers = []
        hidden_dims = config.ansatz_params.get('post_hidden_dims', [64, 32])
        prev_dim = quantum_output_dim
        for hidden_dim in hidden_dims:
            post_layers.append(nn.Linear(prev_dim, hidden_dim))
            post_layers.append(nn.ReLU())
            post_layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        post_layers.append(nn.Linear(prev_dim, 1))
        post_layers.append(nn.Sigmoid())
        self.classical_nn = nn.Sequential(*post_layers)

        self._init_weights()

        print(f"\nArchitechture")
        print(f"Input dimension: {input_dim}")
        if config.use_dim_reduction:
            print(f"Dimension reduction: {input_dim} → {config.total_processed_dim}")
        print(f"Split strategy: {config.split_strategy}")
        print(f"Number of parallel circuits: {config.num_parts}")
        print(f"Features per part: {config.features_per_part}")
        print(f"Quantum circuits: {config.num_parts} × {config.n_qubits} qubits")
        print(f"Encoding: {config.encoding_type}")
        print(f"Ansatz: {config.ansatz_type}")
        print(f"Measurement: {config.measurement_type}")
        print(f"Aggregation: {config.aggregation}")
        print(f"Post-NN: {quantum_output_dim} → {hidden_dims} → 1")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.dim_reduction(x)
        x = self.parallel_quantum(x)
        return self.classical_nn(x)
class ParallelQuantumTrainer:

    def __init__(self, model: ParallelHybridQuantumClassifier, config: ParallelQuantumConfig):
        self.model = model
        self.config = config
        self.stats = ParallelModelStats()
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.criterion = nn.BCELoss()
        self.best_model_state = None
        self.best_val_acc = 0.0
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.5)

    def train_epoch(self, X_train, y_train):
        self.model.train()
        total_loss = 0
        correct = 0
        all_preds = []
        all_labels = []

        indices = torch.randperm(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, len(X_shuffled), self.config.batch_size):
            batch_X = X_shuffled[i:i+self.config.batch_size]
            batch_y = y_shuffled[i:i+self.config.batch_size].unsqueeze(1)

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_y.detach().cpu().numpy())

        avg_loss = total_loss / max(1, len(X_shuffled) / self.config.batch_size)
        accuracy = correct / len(X_shuffled)
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

        return avg_loss, accuracy, auc

    def validate(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self.criterion(outputs, y_val.unsqueeze(1))
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y_val.unsqueeze(1)).sum().item() / len(y_val)
            y_val_cpu = y_val.cpu().numpy()
            outputs_cpu = outputs.cpu().numpy()
            auc = roc_auc_score(y_val_cpu, outputs_cpu)
        return loss.item(), accuracy, auc

    def train(self, X_train, y_train, X_val=None, y_val=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)

        self.model = self.model.to(device)

        self.stats.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.stats.trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nTraining Parallel Quantum Classifier")
        print(f"Parameters: {self.stats.trainable_parameters:,} trainable")
        print(f"Device: {device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            train_loss, train_acc, train_auc = self.train_epoch(X_train_tensor, y_train_tensor)
            self.stats.train_loss.append(train_loss)
            self.stats.train_acc.append(train_acc)
            self.stats.train_auc.append(train_auc)

            if X_val is not None:
                val_loss, val_acc, val_auc = self.validate(X_val_tensor, y_val_tensor)
                self.stats.val_loss.append(val_loss)
                self.stats.val_acc.append(val_acc)
                self.stats.val_auc.append(val_auc)

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.stats.best_val_acc = val_acc
                    self.stats.best_val_auc = val_auc
                    self.stats.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())

                self.scheduler.step(val_acc)

            epoch_time = time.time() - epoch_start
            self.stats.epoch_times.append(epoch_time)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log = f"Epoch {epoch+1:3d}/{self.config.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}"
                if X_val is not None:
                    log += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
                    if val_acc == self.best_val_acc:
                        log += " -> BEST"
                print(log)

        self.stats.total_train_time = time.time() - start_time
        print(f"\nTraining complete! Time: {self.stats.total_train_time:.1f}s")
        if X_val is not None:
            print(f"Best val acc: {self.stats.best_val_acc:.4f} (epoch {self.stats.best_epoch+1})")

        return self.stats

    def predict(self, X_test, use_best_model=True):
        if use_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = (outputs > 0.5).float().cpu().numpy().flatten()
            probabilities = outputs.cpu().numpy().flatten()

        return predictions, probabilities

    def evaluate(self, X_test, y_test, use_best_model=True):
        y_pred, y_prob = self.predict(X_test, use_best_model)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\ntest results")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"Confusion Matrix:")
        print(f"            Predicted")
        print(f"           Neg   Pos")
        print(f"Actual Neg  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       Pos  {cm[1,0]:5d}  {cm[1,1]:5d}")

        return {'accuracy': acc, 'f1_score': f1, 'auc': auc, 'confusion_matrix': cm}

    def plot_history(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.stats.train_loss, label='Train')
        if self.stats.val_loss:
            axes[0].plot(self.stats.val_loss, label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.stats.train_acc, label='Train')
        if self.stats.val_acc:
            axes[1].plot(self.stats.val_acc, label='Val')
            if self.stats.best_epoch >= 0:
                axes[1].axvline(x=self.stats.best_epoch, color='green', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.stats.train_auc, label='Train')
        if self.stats.val_auc:
            axes[2].plot(self.stats.val_auc, label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
@dataclass
class ParallelModelStats:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_auc: List[float] = field(default_factory=list)
    val_auc: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    total_train_time: float = 0.0
    memory_usage: float = 0.0
    parameter_count: int = 0
    trainable_parameters: int = 0
    best_val_acc: float = 0.0
    best_val_auc: float = 0.0
    best_epoch: int = -1

def get_training_info(trainer, model, model_path='parallel_quantum_model.pth'):
    training_duration_seconds = trainer.stats.total_train_time
    training_duration_minutes = training_duration_seconds / 60

    if os.path.exists(model_path):
        storage_bytes = os.path.getsize(model_path)
        storage_mb = storage_bytes / (1024 * 1024)
    else:
        storage_mb = 0

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = (total_params * 4) / (1024 * 1024)

    return {
        'training_duration_seconds': training_duration_seconds,
        'training_duration_minutes': training_duration_minutes,
        'model_storage_mb': storage_mb,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_mb': param_memory_mb,
        'best_epoch': trainer.stats.best_epoch,
        'best_val_acc': trainer.stats.best_val_acc
    }