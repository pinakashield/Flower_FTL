import subprocess
import time
import os
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt

def run_server():
    print("🚀 Starting Flower server...")
    return subprocess.Popen(["python3", "server.py"])  # Updated to use python3

def run_client(client_id):
    print(f"🤖 Starting client {client_id}...")
    return subprocess.Popen(["python3", "client.py", str(client_id)])  # Updated to use python3

def visualize_logs(filepath):
    print("\n📊 Visualizing suspicious client log...")
    if not os.path.exists(filepath):
        print("⚠️ Log file not found.")
        return

    df = pd.read_csv(filepath)

    # Plot number of flagged clients per round
    flagged = df[df['behavior'] != 'Normal']
    round_counts = flagged.groupby('round')['client_id'].count()
    plt.figure(figsize=(10, 4))
    round_counts.plot(kind='bar', color='tomato')
    plt.title("🚨 Flagged Clients Per Round")
    plt.xlabel("Round")
    plt.ylabel("Number of Clients")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot update norm and cosine similarity trends
    plt.figure(figsize=(10, 4))
    for cid in df['client_id'].unique():
        client_data = df[df['client_id'] == cid]
        plt.plot(client_data['round'], client_data['update_norm'], label=f"Client {cid} Norm")
    plt.title("📏 Update Norm Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Update Norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for cid in df['client_id'].unique():
        client_data = df[df['client_id'] == cid]
        plt.plot(client_data['round'], client_data['cosine_similarity'], label=f"Client {cid} CosSim")
    plt.title("📐 Cosine Similarity Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    log_path = "suspicious_clients_log.csv"

    if os.path.exists(log_path):
        os.remove(log_path)

    def run_demo():
        server_proc = multiprocessing.Process(target=run_server)
        server_proc.start()
        time.sleep(3)

        client0 = multiprocessing.Process(target=run_client, args=(0,))  # DDoS
        client1 = multiprocessing.Process(target=run_client, args=(1,))  # Normal

        client0.start()
        client1.start()

        time.sleep(60)

        client0.terminate()
        client1.terminate()
        server_proc.terminate()

    run_demo()

    print("\n✅ Demo completed.")
    visualize_logs(log_path)

if __name__ == "__main__":
    main()
