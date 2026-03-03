import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from scipy.signal import butter, filtfilt




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True, help="Input Data directory")
    parser.add_argument("-out_dir", required=True, help="Output Dataset directory")
    return parser.parse_args()




def read_signal_file(filepath):
    df = pd.read_csv(filepath, skiprows=7, header=None, sep=";")
    df.columns = ["Timestamp", "Value"]

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%d.%m.%Y %H:%M:%S,%f"
    )

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    return df




def read_event_file(filepath):

    events_df = pd.read_csv(
        filepath,
        sep=";",
        skiprows=5,
        header=None,
        names=["TimeRange", "Duration", "EventType", "Stage"]
    )

    events_df = events_df.apply(
        lambda col: col.str.strip() if col.dtype == "object" else col
    )

    events_df[["StartRaw", "EndRaw"]] = events_df["TimeRange"].str.split("-", expand=True)

    events_df["DatePart"] = events_df["StartRaw"].str.split(" ").str[0]
    events_df["EndRaw"] = events_df["DatePart"] + " " + events_df["EndRaw"]

    events_df["StartTime"] = pd.to_datetime(
        events_df["StartRaw"],
        format="%d.%m.%Y %H:%M:%S,%f"
    )

    events_df["EndTime"] = pd.to_datetime(
        events_df["EndRaw"],
        format="%d.%m.%Y %H:%M:%S,%f"
    )

    return events_df




def bandpass_filter(signal, fs=32, low=0.17, high=0.4, order=4):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)




def get_label(window_start, window_end, events_df):

    window_duration = (window_end - window_start).total_seconds()

    for _, row in events_df.iterrows():
        overlap_start = max(window_start, row["StartTime"])
        overlap_end = min(window_end, row["EndTime"])

        overlap = (overlap_end - overlap_start).total_seconds()

        if overlap > 0 and overlap / window_duration > 0.5:
            return row["EventType"]

    return "Normal"




def interpolate_spo2(flow_window, spo2_window):

    # Need at least 2 points to interpolate
    if len(spo2_window) < 2:
        return None

    # Convert timestamps to numeric (nanoseconds)
    flow_times = flow_window["Timestamp"].astype(np.int64).values
    spo2_times = spo2_window["Timestamp"].astype(np.int64).values
    spo2_values = spo2_window["Value"].values

    # Remove NaNs before interpolation
    valid_mask = ~np.isnan(spo2_values)
    spo2_times = spo2_times[valid_mask]
    spo2_values = spo2_values[valid_mask]

    if len(spo2_values) < 2:
        return None

    interpolated = np.interp(flow_times, spo2_times, spo2_values)

    return interpolated




def process_participant(participant_path, participant_name):

    print(f"Processing {participant_name}")

    flow_file = None
    thorac_file = None
    spo2_file = None
    events_file = None

    for file in participant_path.glob("*.txt"):
        name = file.name.lower().strip()

        if "flow events" in name:
            events_file = file
        elif "flow" in name:
            flow_file = file
        elif "thorac" in name:
            thorac_file = file
        elif "spo2" in name:
            spo2_file = file

    if not all([flow_file, thorac_file, spo2_file, events_file]):
        raise FileNotFoundError(f"Missing files in {participant_name}")

    flow_txt = read_signal_file(flow_file)
    thorac_txt = read_signal_file(thorac_file)
    spo2_txt = read_signal_file(spo2_file)
    events_df = read_event_file(events_file)

    # Filter respiration signals
    flow_txt["Value"] = bandpass_filter(flow_txt["Value"].values, fs=32)
    thorac_txt["Value"] = bandpass_filter(thorac_txt["Value"].values, fs=32)

    window_seconds = 30
    overlap = 0.5
    step_seconds = window_seconds * (1 - overlap)

    step = pd.Timedelta(seconds=step_seconds)
    window_duration = pd.Timedelta(seconds=window_seconds)

    dataset = []

    recording_start = flow_txt["Timestamp"].iloc[0]
    recording_end = flow_txt["Timestamp"].iloc[-1]

    current_start = recording_start

    while current_start + window_duration <= recording_end:

        current_end = current_start + window_duration

        flow_window = flow_txt[
            (flow_txt["Timestamp"] >= current_start) &
            (flow_txt["Timestamp"] < current_end)
        ]

        thorac_window = thorac_txt[
            (thorac_txt["Timestamp"] >= current_start) &
            (thorac_txt["Timestamp"] < current_end)
        ]

        spo2_window = spo2_txt[
            (spo2_txt["Timestamp"] >= current_start) &
            (spo2_txt["Timestamp"] < current_end)
        ]

        # Require full 30s at 32Hz → 960 samples
        if len(flow_window) != 960 or len(thorac_window) != 960:
            current_start += step
            continue

        spo2_interp = interpolate_spo2(flow_window, spo2_window)

        if spo2_interp is None or len(spo2_interp) != 960:
            current_start += step
            continue

        if np.isnan(spo2_interp).any():
            current_start += step
            continue

        signal = np.vstack([
            flow_window["Value"].values,
            thorac_window["Value"].values,
            spo2_interp
        ])

        label = get_label(current_start, current_end, events_df)

        dataset.append({
            "participant": participant_name,
            "signal": signal.astype(np.float32),
            "label": label
        })

        current_start += step

    return dataset



def main():

    args = parse_arguments()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    all_data = []

    #Looping over each participat
    for participant_folder in in_dir.iterdir():
        if participant_folder.is_dir():
            participant_name = participant_folder.name
            participant_data = process_participant(
                participant_folder,
                participant_name
            )
            all_data.extend(participant_data)

    with open(out_dir / "breathing_dataset.pkl", "wb") as f:
        pickle.dump(all_data, f)

    print("Dataset creation complete.")
    print(f"Total windows created: {len(all_data)}")


if __name__ == "__main__":
    main()
