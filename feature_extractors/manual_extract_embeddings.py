import xml.etree.ElementTree as ET
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os


class PhysicalMorphologyExtractor:
    """
    Extracts canonical physical parameters that define the dynamics and kinematics
    of a quadruped, regardless of naming convention.
    Focuses on:
    1. Kinematic Chain Lengths (Base->Hip, Hip->Knee, Knee->Foot)
    2. Mass Ratios (Trunk vs Legs)
    3. Normalized Dynamics (Torque/Weight, Inertia distribution)
    4. Geometric Footprint (Stance width/length)
    """
    def __init__(self):
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_names = []

    def parse_usd(self, usd_path: str) -> ET.Element:
        tree = ET.parse(usd_path)
        return tree.getroot()

    def _parse_origin(self, element: Optional[ET.Element]) -> np.ndarray:
        if element is None: return np.zeros(3)
        origin = element.find('origin')
        if origin is not None:
            try:
                return np.array([float(x) for x in origin.get('xyz', '0 0 0').split()])
            except:
                pass
        return np.zeros(3)

    def _get_link_mass(self, root: ET.Element, link_name: str) -> float:
        for link in root.findall('link'):
            if link.get('name') == link_name:
                inertial = link.find('inertial')
                if inertial is not None:
                    mass = inertial.find('mass')
                    if mass is not None:
                        return float(mass.get('value', 0.0))
        return 0.0

    def extract_features(self, usd_path: str) -> Dict[str, float]:
        root = self.parse_usd(usd_path)
        features = {}
        joints = root.findall('joint')
        links = root.findall('link')

        masses = []
        for l in links:
            m = self._get_link_mass(root, l.get('name'))
            masses.append(m)

        total_mass = sum(masses)
        trunk_mass = max(masses) if masses else 0.0

        features['dyn_log_total_mass'] = np.log1p(total_mass)
        features['dyn_trunk_mass_ratio'] = trunk_mass / (total_mass + 1e-6)

        topology = {}
        for joint in joints:
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            topology[child] = {'parent': parent, 'joint': joint, 'name': joint.get('name')}

        all_link_names = set(l.get('name') for l in links)
        child_links = set(topology.keys())
        base_candidates = list(all_link_names - child_links)
        base_link = base_candidates[0] if base_candidates else all_link_names[0]

        # We only want joints attached to base that act as Hip/Leg roots.
        # Filter out Heads, Lidars, IMUs, etc.
        candidates = [child for child, data in topology.items() if data['parent'] == base_link]
        leg_keywords = ['hip', 'thigh', 'leg', 'fl_', 'fr_', 'rl_', 'rr_', 'lf_', 'rf_', 'lh_', 'rh_', 'haa', 'hx']

        leg_roots = []
        for child in candidates:
            joint_name = topology[child]['name'].lower()
            link_name = child.lower()

            # Check 1: Keyword match
            if any(k in joint_name or k in link_name for k in leg_keywords):
                # Check 2: Exclude common accessories even if they have weird names
                if 'head' not in link_name and 'lidar' not in link_name and 'imu' not in link_name:
                    leg_roots.append(child)

        # Fallback: if strict filter removed everything (unlikely), revert to all children
        if not leg_roots:
            leg_roots = candidates

        # Heuristic: Pick Front-Left leg for length measurements
        fl_keywords = ['FL', 'LF', 'front_left', 'left_front']
        chosen_leg = leg_roots[0] if leg_roots else None
        for root_link in leg_roots:
            if any(k.lower() in topology[root_link]['name'].lower() for k in fl_keywords):
                chosen_leg = root_link
                break

        if chosen_leg:
            features['kin_hip_offset'] = np.linalg.norm(self._parse_origin(topology[chosen_leg]['joint']))
        else:
            features['kin_hip_offset'] = 0.0

        # Leg Segments
        segment_lengths = []
        actuator_efforts = []
        current_link = chosen_leg

        for _ in range(3):
            next_link = None
            for child, data in topology.items():
                if data['parent'] == current_link and 'sensor' not in child.lower():
                    next_link = child
                    break

            if next_link:
                joint = topology[next_link]['joint']
                dist = np.linalg.norm(self._parse_origin(joint))
                if dist > 0.01: segment_lengths.append(dist)

                limit = joint.find('limit')
                if limit is not None: actuator_efforts.append(float(limit.get('effort', 0)))
                current_link = next_link
            else:
                break

        features['kin_thigh_length'] = segment_lengths[0] if len(segment_lengths) > 0 else 0.0
        features['kin_shank_length'] = segment_lengths[1] if len(segment_lengths) > 1 else 0.0

        if features['kin_thigh_length'] > 0:
            features['kin_leg_ratio'] = features['kin_shank_length'] / features['kin_thigh_length']
        else:
            features['kin_leg_ratio'] = 0.0

        # Only use the verified leg roots to calculate stance
        hip_positions = []
        for child in leg_roots:
            pos = self._parse_origin(topology[child]['joint'])
            hip_positions.append(pos)

        if hip_positions:
            hip_positions = np.array(hip_positions)
            l = np.ptp(hip_positions[:, 0])
            w = np.ptp(hip_positions[:, 1])
            features['geo_stance_length'] = l
            features['geo_stance_width'] = w
            features['geo_aspect_ratio'] = l / (w + 1e-6)
        else:
            features['geo_stance_length'] = 0.0
            features['geo_stance_width'] = 0.0
            features['geo_aspect_ratio'] = 0.0

        if actuator_efforts and total_mass > 0:
            features['dyn_torque_density'] = np.mean(actuator_efforts) / (total_mass * 9.81)
        else:
            features['dyn_torque_density'] = 0.0

        return features

    def process_directory(self, usd_dir: str) -> Tuple[np.ndarray, List[str]]:
        usd_files = [f for f in os.listdir(usd_dir) if f.endswith('.xml') or f.endswith('.usd')]
        usd_files.sort()

        data_list = []
        filenames = []

        for f in usd_files:
            path = os.path.join(usd_dir, f)
            try:
                 data_list.append(self.extract_features(path))
                 filenames.append(f)
            except Exception as e:
                print(f"Skipping {f}: {e}")

        if not data_list: return np.array([]), []

        self.feature_names = sorted(list(data_list[0].keys()))
        matrix = [[d.get(k, 0.0) for k in self.feature_names] for d in data_list]
        X = np.array(matrix)

        std_devs = np.std(X, axis=0)
        valid_indices = np.where(std_devs > 1e-6)[0]

        dropped = [self.feature_names[i] for i in range(len(self.feature_names)) if i not in valid_indices]
        if dropped: print(f"Dropped constant features: {dropped}")

        self.feature_names = [self.feature_names[i] for i in valid_indices]
        X = X[:, valid_indices]

        X_norm = self.scaler.fit_transform(X)
        return X_norm, filenames

    def print_analysis(self, features_norm: np.ndarray, filenames: List[str]):
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.expand_frame_repr", True)

        df = pd.DataFrame(features_norm, columns=self.feature_names, index=filenames)

        print("\n=== NEW Physical Morphology Feature Analysis ===")
        print("Stats (Mean/Std of normalized features - should be ~0.0 / 1.0):")
        print(df.describe().loc[['mean', 'std']])

        print("\n=== Similarity Matrix (New Features) ===")
        sim_matrix = cosine_similarity(features_norm)
        sim_df = pd.DataFrame(sim_matrix, index=filenames, columns=filenames)
        print(sim_df.round(3))


if __name__ == "__main__":
    usd_directory = "../assets/usds/"

    if os.path.exists(usd_directory):
        print("Processing with NEW PhysicalMorphologyExtractor...")

        # Instantiate the new robust class
        extractor = PhysicalMorphologyExtractor()

        features, filenames = extractor.process_directory(usd_directory)

        if len(features) > 0:
            # Run analysis
            extractor.print_analysis(features, filenames)

            # Save the improved features
            save_path = '../assets/usds/usd_physical_features_minmax_1.npz'
            print(f"\nSaving features to {save_path}")
            np.savez_compressed(
                save_path,
                features=features,
                feature_names=extractor.feature_names,
                robots=[f.replace('.usd', '').replace('.xml', '') for f in filenames]
            )

            # Optional: Debug print of raw values for the first robot to sanity check
            for i, f in enumerate(filenames):
                print(f"\nDebug: Raw values for {f} robot:")
                raw_feats = extractor.extract_features(os.path.join(usd_directory, f))
                for k, v in raw_feats.items():
                    print(f"  {k}: {v:.4f}")

    else:
        print(f"Directory {usd_directory} not found.")