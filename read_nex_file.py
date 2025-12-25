import os
import struct
import numpy as np
from tkinter import filedialog, Tk  # For file dialog (equivalent to uigetfile)

def read_nex_file(file_name=""):
    """
    Read a .nex file and return its data in a dictionary structure.
    
    Args:
        file_name (str): Path to the .nex file. If empty, opens a file dialog.
    
    Returns:
        dict: A dictionary containing the .nex file data with the same structure as MATLAB output.
    """
    nex_file = {}

    # File selection dialog if no file_name provided
    if not file_name:
        root = Tk()
        root.withdraw()  # Hide the main window
        file_name = filedialog.askopenfilename(
            filetypes=[("NeuroExplorer files", "*.nex")],
            title="Select a NeuroExplorer file"
        )
        root.destroy()
        if not file_name:
            raise ValueError("No file was selected")

    # Open file in binary read mode with little-endian specification
    try:
        with open(file_name, 'rb') as fid:
            # Read magic number (int32)
            magic = struct.unpack('<i', fid.read(4))[0]
            if magic != 827868494:
                raise ValueError("The file is not a valid .nex file")

            # Read file header
            nex_file['version'] = struct.unpack('<i', fid.read(4))[0]
            comment = fid.read(256).decode('ascii', errors='ignore').rstrip('\x00')
            nex_file['comment'] = comment
            nex_file['freq'] = struct.unpack('<d', fid.read(8))[0]
            nex_file['tbeg'] = struct.unpack('<i', fid.read(4))[0] / nex_file['freq']
            nex_file['tend'] = struct.unpack('<i', fid.read(4))[0] / nex_file['freq']
            nvar = struct.unpack('<i', fid.read(4))[0]

            # Skip 260 bytes of padding
            fid.seek(260, 1)  # 1 = relative to current position

            # Initialize counters and storage
            neuron_count = 0
            event_count = 0
            interval_count = 0
            wave_count = 0
            pop_count = 0
            cont_count = 0
            marker_count = 0

            nex_file['neurons'] = []
            nex_file['events'] = []
            nex_file['intervals'] = []
            nex_file['waves'] = []
            nex_file['popvectors'] = []
            nex_file['contvars'] = []
            nex_file['markers'] = []

            # Read all variables
            for _ in range(nvar):
                # Read variable header
                var_type = struct.unpack('<i', fid.read(4))[0]
                var_version = struct.unpack('<i', fid.read(4))[0]
                name = fid.read(64).decode('ascii', errors='ignore').rstrip('\x00')
                offset = struct.unpack('<i', fid.read(4))[0]
                n = struct.unpack('<i', fid.read(4))[0]
                wire_number = struct.unpack('<i', fid.read(4))[0]
                unit_number = struct.unpack('<i', fid.read(4))[0]
                gain = struct.unpack('<i', fid.read(4))[0]
                filter = struct.unpack('<i', fid.read(4))[0]
                x_pos = struct.unpack('<d', fid.read(8))[0]
                y_pos = struct.unpack('<d', fid.read(8))[0]
                w_frequency = struct.unpack('<d', fid.read(8))[0]
                ad_to_mv = struct.unpack('<d', fid.read(8))[0]
                n_points_wave = struct.unpack('<i', fid.read(4))[0]
                n_markers = struct.unpack('<i', fid.read(4))[0]
                marker_length = struct.unpack('<i', fid.read(4))[0]
                mv_offset = struct.unpack('<d', fid.read(8))[0]

                file_position = fid.tell()

                # Process variable based on type
                if var_type == 0:  # Neuron
                    neuron_count += 1
                    neuron = {
                        'name': name,
                        'varVersion': var_version,
                        'xPos': x_pos,
                        'yPos': y_pos
                    }
                    if var_version > 100:
                        neuron['wireNumber'] = wire_number
                        neuron['unitNumber'] = unit_number
                    else:
                        neuron['wireNumber'] = 0
                        neuron['unitNumber'] = 0
                    fid.seek(offset, 0)  # 0 = from beginning
                    timestamps = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    neuron['timestamps'] = timestamps.tolist()
                    nex_file['neurons'].append(neuron)

                elif var_type == 1:  # Event
                    event_count += 1
                    event = {'name': name, 'varVersion': var_version}
                    fid.seek(offset, 0)
                    timestamps = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    event['timestamps'] = timestamps.tolist()
                    nex_file['events'].append(event)

                elif var_type == 2:  # Interval
                    interval_count += 1
                    interval = {'name': name, 'varVersion': var_version}
                    fid.seek(offset, 0)
                    int_starts = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    int_ends = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    interval['intStarts'] = int_starts.tolist()
                    interval['intEnds'] = int_ends.tolist()
                    nex_file['intervals'].append(interval)

                elif var_type == 3:  # Waveform
                    wave_count += 1
                    wave = {
                        'name': name,
                        'varVersion': var_version,
                        'NPointsWave': n_points_wave,
                        'WFrequency': w_frequency,
                        'ADtoMV': ad_to_mv
                    }
                    if var_version > 100:
                        wave['wireNumber'] = wire_number
                        wave['unitNumber'] = unit_number
                    else:
                        wave['wireNumber'] = 0
                        wave['unitNumber'] = 0
                    wave['MVOffset'] = mv_offset if nex_file['version'] > 104 else 0
                    fid.seek(offset, 0)
                    timestamps = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    wf = np.frombuffer(fid.read(n_points_wave * n * 2), dtype='<i2').reshape(n_points_wave, n)
                    wave['timestamps'] = timestamps.tolist()
                    wave['waveforms'] = (wf * ad_to_mv + wave['MVOffset']).tolist()
                    nex_file['waves'].append(wave)

                elif var_type == 4:  # Population vector
                    pop_count += 1
                    popvector = {'name': name, 'varVersion': var_version}
                    fid.seek(offset, 0)
                    weights = np.frombuffer(fid.read(n * 8), dtype='<f8')
                    popvector['weights'] = weights.tolist()
                    nex_file['popvectors'].append(popvector)

                elif var_type == 5:  # Continuous variable
                    cont_count += 1
                    contvar = {
                        'name': name,
                        'varVersion': var_version,
                        'ADtoMV': ad_to_mv,
                        'ADFrequency': w_frequency
                    }
                    contvar['MVOffset'] = mv_offset if nex_file['version'] > 104 else 0
                    fid.seek(offset, 0)
                    timestamps = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    fragment_starts = np.frombuffer(fid.read(n * 4), dtype='<i4') + 1  # MATLAB 1-based indexing
                    data = np.frombuffer(fid.read(n_points_wave * 2), dtype='<i2') * ad_to_mv + contvar['MVOffset']
                    contvar['timestamps'] = timestamps.tolist()
                    contvar['fragmentStarts'] = fragment_starts.tolist()
                    contvar['data'] = data.tolist()
                    nex_file['contvars'].append(contvar)

                elif var_type == 6:  # Marker
                    marker_count += 1
                    marker = {'name': name, 'varVersion': var_version}
                    fid.seek(offset, 0)
                    timestamps = np.frombuffer(fid.read(n * 4), dtype='<i4') / nex_file['freq']
                    marker['timestamps'] = timestamps.tolist()
                    marker['values'] = []
                    for _ in range(n_markers):
                        marker_name = fid.read(64).decode('ascii', errors='ignore').rstrip('\x00')
                        value_struct = {'name': marker_name, 'strings': []}
                        for _ in range(n):
                            marker_value = fid.read(marker_length).decode('ascii', errors='ignore').rstrip('\x00')
                            value_struct['strings'].append(marker_value)
                        marker['values'].append(value_struct)
                    nex_file['markers'].append(marker)

                else:
                    print(f"Unknown variable type {var_type}")

                # Return to position after header
                fid.seek(file_position, 0)
                fid.read(60)  # Dummy read to skip padding

    except FileNotFoundError:
        raise ValueError("Unable to open file")

    return nex_file

# Example usage
if __name__ == "__main__":
    try:
        nex_data = read_nex_file("example.nex")  # Replace with your file path or leave empty for dialog
        print(nex_data['version'], nex_data['comment'])
    except ValueError as e:
        print(e)