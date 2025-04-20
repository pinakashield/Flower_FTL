import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Accessing path variables
LOG_PATH = os.getenv("LOG_PATH")
GRAPHS_PATH = os.getenv("GRAPHS_PATH")

#Log files#
CLIENT_FTL_LOG = os.getenv("CLIENT_FTL_LOG")
ROUND_METRICS_LOG = os.getenv("ROUND_METRICS_LOG")
SUSPICIOUS_CLIENTS_LOG = os.getenv("SUSPICIOUS_CLIENTS_LOG")

# Load logs
round_metrics = pd.read_csv(LOG_PATH+ROUND_METRICS_LOG)
client_logs = pd.read_csv(LOG_PATH+CLIENT_FTL_LOG)
suspicious_logs = pd.read_csv(LOG_PATH+SUSPICIOUS_CLIENTS_LOG)

# Per-client accuracy drift

def plot_client_accuracy_drift(client_logs):
    grouped = client_logs.groupby("client_id")
    for cid, group in grouped:
        rounds = group["round"]
        ftl_flags = group["ftl_used"]
        acc_values = group["ftl_used"].astype(int) * 100  # FTL assumed better perf (demo only)

        plt.figure()
        plt.plot(rounds, acc_values, marker='o', label=f"Client {cid} Accuracy (proxy)")
        plt.title(f"Accuracy Drift (Proxy) - Client {cid}")
        plt.xlabel("Round")
        plt.ylabel("Accuracy Estimate (%)")
        plt.ylim(0, 110)
        plt.grid(True)
        plt.legend()
        fname = f"client_{cid}_accuracy_drift.png"
        plt.savefig(GRAPHS_PATH+fname)
        print(f"✅ Saved {fname}")

# Timeline of quarantine events

def plot_quarantine_timeline(suspicious_logs):
    fig, ax = plt.subplots()
    colors = {"Normal": "green", "DDoS": "red", "MITM": "orange"}
    for idx, row in suspicious_logs.iterrows():
        try:
            rnd = int(row["round"])
            cid = int(row["client_id"])  # Ensure client_id is an integer
            label = row["behavior"]
            ax.scatter(rnd, cid, color=colors.get(label, "gray"), label=label if idx == 0 else "")
        except ValueError:
            print(f"⚠️ Skipping invalid client_id or round: {row}")
            continue
    ax.set_xlabel("Round")
    ax.set_ylabel("Client ID")
    ax.set_title("🛡️ Quarantine Timeline")
    ax.grid(True)
    plt.savefig(GRAPHS_PATH+"quarantine_timeline.png")
    print("✅ Saved quarantine_timeline.png")

if __name__ == "__main__":
    plot_client_accuracy_drift(client_logs)
    plot_quarantine_timeline(suspicious_logs)
