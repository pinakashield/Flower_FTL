import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Accessing path variables
LOG_PATH = os.getenv("LOG_PATH")
GRAPHS_PATH = os.getenv("GRAPHS_PATH")
# Load training logs
round_metrics = pd.read_csv(LOG_PATH+"ftl_round_metrics.csv")
client_logs = pd.read_csv(LOG_PATH+"ftl_client_log.csv")

# Plot global training performance
def plot_global_training():
    plt.figure()
    plt.plot(round_metrics["round"], round_metrics["avg_loss"], label="Average Loss")
    plt.plot(round_metrics["round"], round_metrics["avg_accuracy"], label="Average Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Metric")
    plt.title("Global Federated Learning Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig(GRAPHS_PATH+"global_training_curve.png")
    print("📊 Saved global_training_curve.png")

# Plot number of FTL vs non-FTL clients
def plot_ftl_client_distribution():
    counts = client_logs["ftl_used"].value_counts().rename(index={True: "FTL", False: "Standard"})
    plt.figure()
    counts.plot(kind="bar", color=["skyblue", "salmon"])
    plt.title("Client Participation: FTL vs Standard")
    plt.ylabel("Number of Reports")
    plt.xticks(rotation=0)
    plt.grid(True, axis="y")
    plt.savefig(GRAPHS_PATH+"ftl_vs_standard_clients.png")
    print("📊 Saved ftl_vs_standard_clients.png")

# Run all visualizations
if __name__ == "__main__":
    plot_global_training()
    plot_ftl_client_distribution()
