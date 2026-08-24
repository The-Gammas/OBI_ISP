import os
import sys
import json
import argparse
from typing import Optional, List, Tuple
import h5py
import numpy as np
import pandas as pd
from caveclient import CAVEclient

VOXEL_RESOLUTION_UM = np.array([0.004, 0.004, 0.040])


def get_cave_client(datastack_name: str = 'minnie65_public', token: Optional[str] = None) -> CAVEclient:
    try:
        if token:
            client = CAVEclient()
            client.auth.save_token(token=token, overwrite=True)
        client = CAVEclient(datastack_name)
        return client
    except Exception as e:
        print(f"error connecting to cave ({datastack_name}): {e}".lower())
        print("to configure token: python main.py auth --token <your_token>")
        raise SystemExit(1)


def fetch_microns_data(
    num_neurons: int = 20,
    cell_type: str = "excitatory",
    population_name: Optional[str] = None,
    root_ids: Optional[List[int]] = None,
    datastack_name: str = "minnie65_public",
    cell_type_table: str = "baylor_log_reg_cell_type_coarse_v1",
    synapse_table: str = "synapses_pni_2",
    token: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = get_cave_client(datastack_name=datastack_name, token=token)

    if root_ids:
        try:
            candidates_df = client.materialize.query_table(
                cell_type_table,
                filter_in_dict={'pt_root_id': root_ids}
            )
            if len(candidates_df) == 0:
                candidates_df = pd.DataFrame({'pt_root_id': root_ids, 'cell_type': cell_type.lower()})
        except Exception:
            candidates_df = pd.DataFrame({'pt_root_id': root_ids, 'cell_type': cell_type.lower()})

        selected_nodes = candidates_df.head(len(root_ids)).copy().reset_index(drop=True)
        if 'pt_position' not in selected_nodes.columns:
            try:
                nuc_df = client.materialize.query_table(
                    'nucleus_detection_v0',
                    filter_in_dict={'pt_root_id': root_ids}
                )
                selected_nodes = selected_nodes.merge(
                    nuc_df[['pt_root_id', 'pt_position']], on='pt_root_id', how='left'
                )
            except Exception:
                selected_nodes['pt_position'] = [[0, 0, 0] for _ in range(len(selected_nodes))]
    else:
        candidates_df = client.materialize.query_table(
            cell_type_table,
            filter_equal_dict={'cell_type': cell_type.lower()},
            limit=max(500, num_neurons * 5)
        )
        if candidates_df.empty:
            raise ValueError(f"no neurons found for cell_type='{cell_type.lower()}' in '{cell_type_table}'.")
        selected_nodes = candidates_df.head(num_neurons).copy().reset_index(drop=True)

    target_root_ids = [int(rid) for rid in selected_nodes['pt_root_id'].tolist()]
    actual_count = len(target_root_ids)

    internal_synapses = client.materialize.query_table(
        synapse_table,
        filter_in_dict={'pre_pt_root_id': target_root_ids, 'post_pt_root_id': target_root_ids}
    )

    total_incoming_synapses = client.materialize.query_table(
        synapse_table,
        filter_in_dict={'post_pt_root_id': target_root_ids}
    )

    internal_in_counts = internal_synapses['post_pt_root_id'].value_counts() if not internal_synapses.empty else pd.Series(dtype=int)
    total_in_counts = total_incoming_synapses['post_pt_root_id'].value_counts() if not total_incoming_synapses.empty else pd.Series(dtype=int)

    audit_records = []
    for root_id in target_root_ids:
        tot = int(total_in_counts.get(root_id, 0))
        inte = int(internal_in_counts.get(root_id, 0))
        ext = max(0, tot - inte)
        ratio = (inte / tot * 100.0) if tot > 0 else 0.0
        audit_records.append({
            'microns_root_id': root_id,
            'total_synapses': tot,
            'internal_synapses': inte,
            'extrinsic_synapses': ext,
            'recurrent_fraction_pct': round(ratio, 2)
        })

    extrinsic_audit_df = pd.DataFrame(audit_records)
    total_internal = len(internal_synapses)
    total_all_inputs = len(total_incoming_synapses)
    total_extrinsic = max(0, total_all_inputs - total_internal)

    print(f"neurons selected: {actual_count}")
    print(f"recurrent links: {total_internal}")
    print(f"extrinsic severed inputs: {total_extrinsic} (average {total_extrinsic / actual_count:.1f} per neuron)")

    return selected_nodes, internal_synapses, extrinsic_audit_df


def convert_microns_to_sonata(
    nodes_df: pd.DataFrame,
    synapses_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    population_name: str = "v1_l23_excitatory"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    id_map = {int(root_id): idx for idx, root_id in enumerate(nodes_df['pt_root_id'])}

    if not nodes_df.empty and 'pt_position' in nodes_df.columns:
        soma_positions_vx = np.vstack(nodes_df['pt_position'].values)
        soma_positions_um = soma_positions_vx * VOXEL_RESOLUTION_UM
    else:
        soma_positions_um = np.zeros((len(nodes_df), 3))

    cell_types = nodes_df['cell_type'].astype(str).str.lower() if 'cell_type' in nodes_df.columns else pd.Series(["excitatory"] * len(nodes_df))

    sonata_nodes = pd.DataFrame({
        'node_id': nodes_df['pt_root_id'].map(id_map).astype(np.int64),
        'node_type_id': 100,
        'microns_root_id': nodes_df['pt_root_id'].astype(np.int64),
        'x': np.round(soma_positions_um[:, 0], 3),
        'y': np.round(soma_positions_um[:, 1], 3),
        'z': np.round(soma_positions_um[:, 2], 3),
        'cell_type': cell_types.values,
        'extrinsic_synapses_count': audit_df['extrinsic_synapses'].values.astype(np.int64)
    })

    if not synapses_df.empty and 'ctr_pt_position' in synapses_df.columns:
        syn_pos_vx = np.vstack(synapses_df['ctr_pt_position'].values)
        syn_pos_um = syn_pos_vx * VOXEL_RESOLUTION_UM
        syn_ids = synapses_df['id'].astype(np.int64) if 'id' in synapses_df.columns else np.arange(len(synapses_df), dtype=np.int64)
        syn_sizes = synapses_df['size'].astype(np.float64) if 'size' in synapses_df.columns else np.ones(len(synapses_df), dtype=np.float64)

        sonata_edges = pd.DataFrame({
            'edge_id': np.arange(len(synapses_df), dtype=np.int64),
            'source_node_id': synapses_df['pre_pt_root_id'].map(id_map).astype(np.int64),
            'target_node_id': synapses_df['post_pt_root_id'].map(id_map).astype(np.int64),
            'edge_type_id': 101,
            'synapse_id_em': syn_ids.values,
            'synapse_size_voxels': syn_sizes.values,
            'syn_pos_x': np.round(syn_pos_um[:, 0], 3),
            'syn_pos_y': np.round(syn_pos_um[:, 1], 3),
            'syn_pos_z': np.round(syn_pos_um[:, 2], 3)
        })
    else:
        sonata_edges = pd.DataFrame(columns=[
            'edge_id', 'source_node_id', 'target_node_id', 'edge_type_id',
            'synapse_id_em', 'synapse_size_voxels', 'syn_pos_x', 'syn_pos_y', 'syn_pos_z'
        ])

    return sonata_nodes, sonata_edges


def write_sonata_circuit_files(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    audit_df: Optional[pd.DataFrame] = None,
    population_name: str = "v1_l23_excitatory",
    output_dir: str = ".",
    export_csv: bool = True
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    created_files = {}

    nodes_h5_path = os.path.join(output_dir, 'nodes.h5')
    with h5py.File(nodes_h5_path, 'w') as f:
        grp = f.create_group(f'/nodes/{population_name}')
        grp.create_dataset('node_id', data=nodes_df['node_id'].values, dtype='i8')
        grp.create_dataset('node_type_id', data=nodes_df['node_type_id'].values, dtype='i8')

        g0 = grp.create_group('0')
        g0.create_dataset('x', data=nodes_df['x'].values, dtype='f8')
        g0.create_dataset('y', data=nodes_df['y'].values, dtype='f8')
        g0.create_dataset('z', data=nodes_df['z'].values, dtype='f8')
        g0.create_dataset('microns_root_id', data=nodes_df['microns_root_id'].values, dtype='i8')
        g0.create_dataset('extrinsic_synapses_count', data=nodes_df['extrinsic_synapses_count'].values, dtype='i8')
    created_files['nodes_h5'] = nodes_h5_path

    node_types_path = os.path.join(output_dir, 'node_types.csv')
    is_inhibitory = "inhibitory" in population_name.lower()
    node_types_df = pd.DataFrame([{
        'node_type_id': 100,
        'pop_name': population_name,
        'model_type': 'biophysical',
        'model_template': 'nrn:l23_bc' if is_inhibitory else 'nrn:l23_pc',
        'morphology': 'l23_basket.swc' if is_inhibitory else 'l23_pyramidal.swc',
        'dynamics_params': 'l23_bc_fit.json' if is_inhibitory else 'l23_pc_fit.json'
    }])
    node_types_df.to_csv(node_types_path, index=False)
    created_files['node_types_csv'] = node_types_path

    edge_pop_name = f"{population_name}_to_{population_name}"
    edges_h5_path = os.path.join(output_dir, 'edges.h5')
    with h5py.File(edges_h5_path, 'w') as f:
        grp = f.create_group(f'/edges/{edge_pop_name}')
        if len(edges_df) > 0:
            grp.create_dataset('source_node_id', data=edges_df['source_node_id'].values, dtype='i8')
            grp.create_dataset('target_node_id', data=edges_df['target_node_id'].values, dtype='i8')
            grp.create_dataset('edge_type_id', data=edges_df['edge_type_id'].values, dtype='i8')

            g0 = grp.create_group('0')
            g0.create_dataset('synapse_size', data=edges_df['synapse_size_voxels'].values, dtype='f8')
            g0.create_dataset('syn_pos_x', data=edges_df['syn_pos_x'].values, dtype='f8')
            g0.create_dataset('syn_pos_y', data=edges_df['syn_pos_y'].values, dtype='f8')
            g0.create_dataset('syn_pos_z', data=edges_df['syn_pos_z'].values, dtype='f8')
        else:
            grp.create_dataset('source_node_id', data=np.array([], dtype='i8'), dtype='i8')
            grp.create_dataset('target_node_id', data=np.array([], dtype='i8'), dtype='i8')
            grp.create_dataset('edge_type_id', data=np.array([], dtype='i8'), dtype='i8')
            g0 = grp.create_group('0')
            g0.create_dataset('synapse_size', data=np.array([], dtype='f8'), dtype='f8')
            g0.create_dataset('syn_pos_x', data=np.array([], dtype='f8'), dtype='f8')
            g0.create_dataset('syn_pos_y', data=np.array([], dtype='f8'), dtype='f8')
            g0.create_dataset('syn_pos_z', data=np.array([], dtype='f8'), dtype='f8')
    created_files['edges_h5'] = edges_h5_path

    edge_types_path = os.path.join(output_dir, 'edge_types.csv')
    edge_types_df = pd.DataFrame([{
        'edge_type_id': 101,
        'model_template': 'exp2syn',
        'delay': 1.5,
        'weight': 0.002
    }])
    edge_types_df.to_csv(edge_types_path, index=False)
    created_files['edge_types_csv'] = edge_types_path

    circuit_config_path = os.path.join(output_dir, 'circuit_config.json')
    circuit_config = {
        "version": "1",
        "manifest": {
            "$base_dir": "."
        },
        "networks": {
            "nodes": [
                {
                    "nodes_file": "$base_dir/nodes.h5",
                    "node_types_file": "$base_dir/node_types.csv"
                }
            ],
            "edges": [
                {
                    "edges_file": "$base_dir/edges.h5",
                    "edge_types_file": "$base_dir/edge_types.csv"
                }
            ]
        }
    }
    with open(circuit_config_path, 'w') as f:
        json.dump(circuit_config, f, indent=2)
    created_files['circuit_config'] = circuit_config_path

    if export_csv:
        nodes_csv_path = os.path.join(output_dir, 'nodes.csv')
        nodes_df.to_csv(nodes_csv_path, index=False)
        created_files['nodes_csv'] = nodes_csv_path

        edges_csv_path = os.path.join(output_dir, 'edges.csv')
        edges_df.to_csv(edges_csv_path, index=False)
        created_files['edges_csv'] = edges_csv_path

        if audit_df is not None:
            audit_csv_path = os.path.join(output_dir, 'synaptic_debt_audit.csv')
            audit_df.to_csv(audit_csv_path, index=False)
            created_files['audit_csv'] = audit_csv_path

        report_path = os.path.join(output_dir, 'summary_report.txt')
        generate_summary_report_file(nodes_df, edges_df, audit_df, population_name, report_path)
        created_files['summary_report'] = report_path

    return created_files


def generate_summary_report_file(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    audit_df: Optional[pd.DataFrame],
    population_name: str,
    output_path: str
):
    num_nodes = len(nodes_df)
    num_edges = len(edges_df)

    if num_nodes > 0:
        x_min, x_max = float(nodes_df['x'].min()), float(nodes_df['x'].max())
        y_min, y_max = float(nodes_df['y'].min()), float(nodes_df['y'].max())
        z_min, z_max = float(nodes_df['z'].min()), float(nodes_df['z'].max())
        span_x, span_y, span_z = x_max - x_min, y_max - y_min, z_max - z_min
    else:
        x_min = x_max = y_min = y_max = z_min = z_max = span_x = span_y = span_z = 0.0

    lines = []
    lines.append(f"circuit data audit report")
    lines.append(f"population: {population_name.lower()} ")
    lines.append("1. population overview")
    lines.append(f"total neurons (nodes): {num_nodes}")
    lines.append(f"recurrent connections (edges): {num_edges}")
    lines.append(f"spatial span (um): x: {span_x:.1f} um | y: {span_y:.1f} um | z: {span_z:.1f} um")
    lines.append(f"soma bounding box (um): [{x_min:.1f}, {y_min:.1f}, {z_min:.1f}] to [{x_max:.1f}, {y_max:.1f}, {z_max:.1f}] ")

    if audit_df is not None and not audit_df.empty:
        total_int = int(audit_df['internal_synapses'].sum())
        total_ext = int(audit_df['extrinsic_synapses'].sum())
        total_all = total_int + total_ext
        pct_int = (total_int / total_all * 100.0) if total_all > 0 else 0.0
        pct_ext = (total_ext / total_all * 100.0) if total_all > 0 else 0.0

        lines.append("2. synaptic connectivity and extrinsic debt audit")
        lines.append(f"total incoming synaptic inputs: {total_all:,}")
        lines.append(f"recurrent inputs (preserved): {total_int:,} ({pct_int:.2f}%)")
        lines.append(f"extrinsic inputs (severed): {total_ext:,} ({pct_ext:.2f}%)")
        lines.append(f"avg extrinsic inputs per neuron: {total_ext / num_nodes:.1f} ")

    if num_edges > 0:
        lines.append("3. synapse morphometrics (cleft size)")
        sizes = edges_df['synapse_size_voxels']
        lines.append(f"min size (voxels): {sizes.min():.1f}")
        lines.append(f"median size (voxels): {sizes.median():.1f}")
        lines.append(f"mean size (voxels): {sizes.mean():.1f}")
        lines.append(f"max size (voxels): {sizes.max():.1f} ")

    with open(output_path, 'w') as f:
        f.write(" ".join(lines).lower() + " ")


def print_data_audit_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, audit_df: pd.DataFrame):
    print(" data points audit summary ")

    print("[a] converted nodes sample:")
    cols_to_show = ['node_id', 'microns_root_id', 'x', 'y', 'z', 'cell_type', 'extrinsic_synapses_count']
    available_cols = [c for c in cols_to_show if c in nodes_df.columns]
    print(nodes_df[available_cols].head(5).to_string(index=False).lower())

    print(" [b] converted edges sample:")
    if not edges_df.empty:
        edge_cols = ['edge_id', 'source_node_id', 'target_node_id', 'synapse_size_voxels', 'syn_pos_x', 'syn_pos_y', 'syn_pos_z']
        avail_edge_cols = [c for c in edge_cols if c in edges_df.columns]
        print(edges_df[avail_edge_cols].head(5).to_string(index=False).lower())
    else:
        print("(no internal recurrent edges found)")

    print(" [c] synaptic debt breakdown:")
    print(audit_df.head(5).to_string(index=False).lower())

    total_int = audit_df['internal_synapses'].sum()
    total_ext = audit_df['extrinsic_synapses'].sum()
    total_all = total_int + total_ext
    pct_int = (total_int / total_all * 100) if total_all > 0 else 0
    pct_ext = (total_ext / total_all * 100) if total_all > 0 else 0

    print(" [d] network connectivity totals:")
    print(f"total synaptic inputs landing on population: {total_all:,}")
    print(f"recurrent (preserved in edges): {total_int:,} ({pct_int:.2f}%)")
    print(f"extrinsic (severed background input debt): {total_ext:,} ({pct_ext:.2f}%)")
    print(f"average extrinsic inputs per neuron: {total_ext / len(nodes_df):.1f} ")


def inspect_dataset(target_dir: str = "."):
    target_dir = os.path.abspath(target_dir).lower()
    print(f" inspecting circuit dataset at: {target_dir} ")

    nodes_h5 = os.path.join(target_dir, "nodes.h5")
    edges_h5 = os.path.join(target_dir, "edges.h5")
    nodes_csv = os.path.join(target_dir, "nodes.csv")
    edges_csv = os.path.join(target_dir, "edges.csv")
    audit_csv = os.path.join(target_dir, "synaptic_debt_audit.csv")
    config_json = os.path.join(target_dir, "circuit_config.json")

    found_files = []

    if os.path.exists(nodes_h5):
        found_files.append("nodes.h5")
        try:
            with h5py.File(nodes_h5, 'r') as f:
                pop_keys = list(f['nodes'].keys()) if 'nodes' in f else []
                print(f"nodes.h5 (populations: {[str(p).lower() for p in pop_keys]})")
                for pop in pop_keys:
                    grp = f[f'nodes/{pop}']
                    node_ids = grp['node_id'][:]
                    print(f"population '{str(pop).lower()}': {len(node_ids)} neurons")
                    if '0' in grp:
                        g0 = grp['0']
                        if 'x' in g0 and 'y' in g0 and 'z' in g0:
                            print(f"x span: [{g0['x'][:].min():.1f}, {g0['x'][:].max():.1f}] um")
                            print(f"y span: [{g0['y'][:].min():.1f}, {g0['y'][:].max():.1f}] um")
                            print(f"z span: [{g0['z'][:].min():.1f}, {g0['z'][:].max():.1f}] um")
        except Exception as e:
            print(f"error reading nodes.h5: {e}".lower())

    if os.path.exists(edges_h5):
        found_files.append("edges.h5")
        try:
            with h5py.File(edges_h5, 'r') as f:
                edge_keys = list(f['edges'].keys()) if 'edges' in f else []
                print(f" edges.h5 (projections: {[str(p).lower() for p in edge_keys]})")
                for edge_pop in edge_keys:
                    grp = f[f'edges/{edge_pop}']
                    sources = grp['source_node_id'][:]
                    print(f"projection '{str(edge_pop).lower()}': {len(sources)} directed synapses")
                    if '0' in grp:
                        g0 = grp['0']
                        if 'synapse_size' in g0 and len(g0['synapse_size']) > 0:
                            print(f"synapse size (voxels): min={g0['synapse_size'][:].min():.1f}, median={np.median(g0['synapse_size'][:]):.1f}, max={g0['synapse_size'][:].max():.1f}")
        except Exception as e:
            print(f"error reading edges.h5: {e}".lower())

    if os.path.exists(nodes_csv):
        found_files.append("nodes.csv")
        df = pd.read_csv(nodes_csv)
        print(f" nodes.csv: {len(df)} rows | columns: {[str(c).lower() for c in df.columns]}")

    if os.path.exists(edges_csv):
        found_files.append("edges.csv")
        df = pd.read_csv(edges_csv)
        print(f"edges.csv: {len(df)} rows | columns: {[str(c).lower() for c in df.columns]}")

    if os.path.exists(audit_csv):
        found_files.append("synaptic_debt_audit.csv")
        df = pd.read_csv(audit_csv)
        total_int = df['internal_synapses'].sum()
        total_ext = df['extrinsic_synapses'].sum()
        total_all = total_int + total_ext
        pct_int = (total_int / total_all * 100) if total_all > 0 else 0
        print(f"synaptic_debt_audit.csv: {len(df)} neurons | recurrent: {total_int} ({pct_int:.2f}%) | severed extrinsic: {total_ext}")

    if os.path.exists(config_json):
        found_files.append("circuit_config.json")
        print(f"circuit_config.json: present")

    if not found_files:
        print(f"no circuit files found in {target_dir}")
    print()


def print_tables_and_cell_types(datastack_name: str = "minnie65_public"):
    print(f" cave dataset: {datastack_name.lower()} ")

    try:
        client = get_cave_client(datastack_name)
        tables = client.materialize.get_tables()
    except Exception:
        tables = [
            'baylor_log_reg_cell_type_coarse_v1',
            'baylor_gnn_cell_type_fine_model_v2',
            'aibs_metamodel_celltypes_v661',
            'allen_column_mtypes_v2',
            'synapses_pni_2',
            'nucleus_detection_v0'
        ]

    print("available tables:")
    for idx, tbl in enumerate(tables, 1):
        print(f"{idx:2d}. {str(tbl).lower()}")

    print(" cell type classification tables & supported cell types:")
    cell_type_info = {
        "baylor_log_reg_cell_type_coarse_v1": [
            "excitatory",
            "inhibitory"
        ],
        "baylor_gnn_cell_type_fine_model_v2": [
            "23p",
            "4p",
            "5p-pt",
            "6p-it",
            "6p-ct",
            "bc",
            "bpc",
            "mc",
            "ngc"
        ],
        "aibs_metamodel_celltypes_v661": [
            "23p",
            "bc",
            "bpc",
            "mc",
            "ngc",
            "astrocyte",
            "microglia",
            "oligo",
            "pericyte"
        ],
        "allen_column_mtypes_v2": [
            "l2a",
            "l2b",
            "l2c",
            "l3a",
            "l3b",
            "l4a",
            "l5b",
            "l6tall-b",
            "itc",
            "dtc"
        ]
    }

    for tbl, types in cell_type_info.items():
        print(f"table: {tbl.lower()}")
        print(f"cell types: {', '.join(t.lower() for t in types)} ")

    print("synapse tables:")
    print("synapses_pni_2 (default synapse table)")
    print("synapse_target_predictions_ssa_v2")
    print("synapse_spine_mapping_v2 ")


def handle_auth_command(token: Optional[str] = None, check_only: bool = False, datastack_name: str = "minnie65_public"):
    print(" cave authentication ")

    client = CAVEclient()
    if token:
        client.auth.save_token(token=token, overwrite=True)
        print("token saved successfully.")

    current_token = client.auth.token
    if current_token:
        masked_token = current_token[:6] + "..." + current_token[-4:] if len(current_token) > 10 else "***"
        print(f"stored token: {masked_token.lower()}")
        try:
            test_client = CAVEclient(datastack_name)
            tables = test_client.materialize.get_tables()
            print(f"connection successful to '{datastack_name.lower()}'. available tables: {len(tables)}")
        except Exception as e:
            print(f"connection test failed: {e}".lower())
    else:
        print("no cave token found.")
        print("1. get token: https://global.daf-apis.com/auth/api/v1/user/token")
        print("2. save: python main.py auth --token <token>")
    print()


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="microns data extractor & sonata circuit generator cli",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="subcommand (default: fetch)")

    fetch_parser = subparsers.add_parser("fetch", help="fetch data from cave and generate sonata files")
    for p in [parser, fetch_parser]:
        p.add_argument("-n", "--num-neurons", type=int, default=20, help="number of neurons to extract")
        p.add_argument("-c", "--cell-type", type=str, default="excitatory", help="cell type filter")
        p.add_argument("--root-ids", type=str, default=None, help="comma-separated microns root ids")
        p.add_argument("-p", "--population-name", type=str, default=None, help="sonata population name")
        p.add_argument("-o", "--output-dir", type=str, default=".", help="output directory")
        p.add_argument("-d", "--dataset", type=str, default="minnie65_public", help="cave datastack name")
        p.add_argument("--cell-type-table", type=str, default="baylor_log_reg_cell_type_coarse_v1", help="cell type table")
        p.add_argument("--synapse-table", type=str, default="synapses_pni_2", help="synapse table")
        p.add_argument("--no-csv", dest="export_csv", action="store_false", default=True, help="disable csv exports")

    inspect_parser = subparsers.add_parser("inspect", help="inspect existing circuit files")
    inspect_parser.add_argument("--dir", "-d", type=str, default=".", help="directory to inspect")

    subparsers.add_parser("explain", help="list available cave tables and cell types")

    auth_parser = subparsers.add_parser("auth", help="check or set cave auth token")
    auth_parser.add_argument("--token", "-t", type=str, default=None, help="set cave token")
    auth_parser.add_argument("--check", action="store_true", default=False, help="check token status")
    auth_parser.add_argument("--dataset", type=str, default="minnie65_public", help="datastack name to test")

    return parser


def run_pipeline(
    num_neurons: int = 20,
    cell_type: str = "excitatory",
    population_name: Optional[str] = None,
    root_ids: Optional[List[int]] = None,
    output_dir: str = ".",
    datastack_name: str = "minnie65_public",
    cell_type_table: str = "baylor_log_reg_cell_type_coarse_v1",
    synapse_table: str = "synapses_pni_2",
    export_csv: bool = True
):
    if population_name is None:
        population_name = f"v1_l23_{cell_type.lower()}"

    raw_nodes, raw_synapses, audit_df = fetch_microns_data(
        num_neurons=num_neurons,
        cell_type=cell_type,
        population_name=population_name,
        root_ids=root_ids,
        datastack_name=datastack_name,
        cell_type_table=cell_type_table,
        synapse_table=synapse_table
    )

    sonata_nodes, sonata_edges = convert_microns_to_sonata(
        raw_nodes, raw_synapses, audit_df, population_name=population_name
    )

    write_sonata_circuit_files(
        sonata_nodes, sonata_edges, audit_df=audit_df,
        population_name=population_name, output_dir=output_dir, export_csv=export_csv
    )

    print_data_audit_summary(sonata_nodes, sonata_edges, audit_df)


def main():
    parser = build_cli_parser()
    args, unknown = parser.parse_known_args()

    if args.command == "explain":
        dataset = getattr(args, 'dataset', 'minnie65_public')
        print_tables_and_cell_types(datastack_name=dataset)
        return

    if args.command == "inspect":
        inspect_dataset(target_dir=args.dir)
        return

    if args.command == "auth":
        handle_auth_command(token=args.token, check_only=args.check, datastack_name=args.dataset)
        return

    root_ids_list = None
    if args.root_ids:
        try:
            root_ids_list = [int(rid.strip()) for rid in args.root_ids.split(",") if rid.strip()]
        except ValueError:
            print("error: --root-ids must be comma-separated integers.")
            sys.exit(1)

    run_pipeline(
        num_neurons=args.num_neurons,
        cell_type=args.cell_type,
        population_name=args.population_name,
        root_ids=root_ids_list,
        output_dir=args.output_dir,
        datastack_name=args.dataset,
        cell_type_table=args.cell_type_table,
        synapse_table=args.synapse_table,
        export_csv=args.export_csv
    )


if __name__ == "__main__":
    main()
