import h5py
import pandas as pd

with h5py.File('nodes.h5', 'r') as f:
    print('keys in nodes.h5 file:', list(f['nodes/V1_L23_excitatory'].keys()))

    pop = f['nodes/V1_L23_excitatory']
    print(pop['0'].keys())
    nodes_df = pd.DataFrame({
        'node_id': pop['node_id'][:],
        'node_type_id': pop['node_type_id'][:],
        'x': pop['0']['x'][:],
        'y': pop['0']['y'][:],
        'z': pop['0']['z'][:],
        'extrinsic_synapses_count': pop['0']['extrinsic_synapses_count'][:]

    })

    print(nodes_df.head())