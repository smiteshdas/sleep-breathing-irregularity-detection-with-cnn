import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-name", required=True, help="Participant folder path")
args = parser.parse_args()

participant_folder = Path(args.name)
participant_name = participant_folder.name

flow_file = None
events_file = None
thorac_file = None
spo2_file = None

for file in participant_folder.glob("*.txt"):

    name = file.name.lower().strip()

    if "flow events" in name:
        events_file = file
    elif "flow" in name:
        flow_file = file
    elif "thorac" in name:
        thorac_file = file
    elif "spo2" in name:
        spo2_file = file

if not all([flow_file, events_file, thorac_file, spo2_file]):
    raise FileNotFoundError("One or more required signal files are missing.")


print("Participant:", participant_name)
print("Flow:", flow_file)
print("Flow Events:", events_file)
print("Thoracic:", thorac_file)
print("SpO2:", spo2_file)

# files ke paths mil gaye

flow_txt= pd.read_csv(flow_file, skiprows=7, header=None, sep=";")
flow_txt.columns = ["Timestamp", "Value"]
flow_txt['Timestamp'] = pd.to_datetime(flow_txt['Timestamp'], format="%d.%m.%Y %H:%M:%S,%f")

thorac_txt= pd.read_csv(thorac_file, skiprows=7, header=None, sep=";")
thorac_txt.columns = ["Timestamp", "Value"]
thorac_txt['Timestamp'] = pd.to_datetime(thorac_txt['Timestamp'], format="%d.%m.%Y %H:%M:%S,%f")

spo2_txt= pd.read_csv(spo2_file, skiprows=7, header=None, sep=";")
spo2_txt.columns = ["Timestamp", "Value"]
spo2_txt['Timestamp'] = pd.to_datetime(spo2_txt['Timestamp'], format="%d.%m.%Y %H:%M:%S,%f")

events_df = pd.read_csv(events_file,sep=";",skiprows=5,header=None,names=["TimeRange", "Duration", "EventType", "Stage"])

events_df = events_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)




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

#files dataframe me badal diya with datetime processed


window_seconds = 300   # 5 mins


recording_start = flow_txt["Timestamp"].iloc[0]
recording_end   = flow_txt["Timestamp"].iloc[-1]

with PdfPages(f"Visualizations/{participant_name}_visualization.pdf") as pdf:

    current_start = recording_start

    while current_start < recording_end:

        proposed_end = current_start + pd.Timedelta(seconds=window_seconds)
        current_end = min(proposed_end, recording_end)

        
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            sharex=True,
            figsize=(18, 6)
        )

        

        ax1.plot(flow_txt["Timestamp"], flow_txt["Value"])
        ax1.set_ylabel("Nasal Airflow")
        ax1.tick_params(labelbottom=False)
        ax1.grid(True)

        ax2.plot(thorac_txt["Timestamp"], thorac_txt["Value"], color='orange')
        ax2.set_ylabel("Thoracic Movement")
        ax2.tick_params(labelbottom=False)
        ax2.grid(True)

        ax3.plot(spo2_txt["Timestamp"], spo2_txt["Value"], color='purple')
        ax3.set_ylabel("SpO2")
        ax3.set_xlabel("Time")
        ax3.grid(True)

        
        ax3.set_xlim(current_start, current_end)

        
        def get_range(df):
            temp = df[(df["Timestamp"] >= current_start) & (df["Timestamp"] <= current_end)]
            return temp["Value"].min(), temp["Value"].max()

        def add_padding(min_val, max_val, percent=0.1):
            if pd.isna(min_val) or pd.isna(max_val):
                return -1, 1
            if min_val == max_val:
                min_val -= 1
                max_val += 1
            padding = (max_val - min_val) * percent
            return min_val - padding, max_val + padding

        ax1.set_ylim(add_padding(*get_range(flow_txt)))
        ax2.set_ylim(add_padding(*get_range(thorac_txt)))
        ax3.set_ylim(add_padding(*get_range(spo2_txt)))

        
        locator = mdates.SecondLocator(interval=5)
        formatter = mdates.DateFormatter('%d %H:%M:%S')

        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(formatter)
        ax3.tick_params(axis='x', rotation=90)

        
        visible_events = events_df[
            (events_df["StartTime"] <= current_end) &
            (events_df["EndTime"] >= current_start)
        ]

        color_map = {
            "Obstructive Apnea": "red",
            "Hypopnea": "yellow"
        }

        for _, row in visible_events.iterrows():
            event_type = row["EventType"].strip()

            if event_type in color_map:
                ax1.axvspan(
                    row["StartTime"],
                    row["EndTime"],
                    alpha=0.3,
                    color=color_map[event_type]
                )

                midpoint = row["StartTime"] + (row["EndTime"] - row["StartTime"]) / 2
                y_top = ax1.get_ylim()[1]

                ax1.text(
                    midpoint,
                    y_top * 0.9,
                    event_type,
                    fontsize=8,
                    ha="center",
                    va="top"
                )

        
        plt.suptitle(f"{participant_name} | {current_start} to {current_end}")

        plt.tight_layout()

        pdf.savefig(fig)   
        plt.close(fig)     

        
        current_start = current_end

print("Visualization PDF created successfully")