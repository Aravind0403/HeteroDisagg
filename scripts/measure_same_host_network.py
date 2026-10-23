#!/usr/bin/env python3
"""
measure_same_host_network.py
Automated end-to-end benchmarking for same-host heterogeneous cluster:
  Prefill Node (A100 SXM4) <---> Decode Node (RTX 3090)
Provisioned under Host 399360 (California, US) on the same layer-2 subnet (192.168.100.0/24).
"""

import subprocess
import json
import time
import sys
import os
import re

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

def run_ssh(host, port, cmd, timeout=30):
    ssh_cmd = [
        'ssh', '-i', SSH_KEY,
        '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'ConnectTimeout=10',
        '-p', str(port),
        f'root@{host}',
        cmd
    ]
    res = subprocess.run(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr

def get_instances():
    raw = subprocess.check_output(['vastai', 'show', 'instances', '--raw'])
    instances = json.loads(raw)
    prefill = next((i for i in instances if 'prefill' in i.get('label', '')), None)
    decode = next((i for i in instances if 'decode' in i.get('label', '')), None)
    return prefill, decode

def parse_local_ip(inst):
    ips = (inst.get('local_ipaddrs') or '').split()
    for ip in ips:
        if ip.startswith('192.168.100.'):
            return ip
    return ips[0] if ips else None

def main():
    print("=" * 70)
    print("HETERODISAGG SAME-HOST EMPIRICAL VALIDATION HARNESS")
    print("=" * 70)

    prefill, decode = get_instances()
    if not prefill or not decode:
        print("Error: Could not find both prefill and decode instances.")
        sys.exit(1)

    print(f"Prefill: ID {prefill['id']} | {prefill['gpu_name']} | Host {prefill['host_id']}")
    print(f"Decode:  ID {decode['id']} | {decode['gpu_name']} | Host {decode['host_id']}")

    prefill_ip = parse_local_ip(prefill)
    decode_ip = parse_local_ip(decode)
    print(f"Internal Subnet IPs: Prefill={prefill_ip} <---> Decode={decode_ip}")

    # Verify Decode SSH
    decode_direct_port = decode.get('machine_dir_ssh_port')
    decode_pub_ip = decode.get('public_ipaddr')
    print(f"\nChecking SSH to Decode Node ({decode_pub_ip}:{decode_direct_port})...")
    rc, out, err = run_ssh(decode_pub_ip, decode_direct_port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if rc != 0:
        print(f"Decode Node not ready yet (rc={rc}): {err.strip()}")
        sys.exit(1)
    print(f"Decode Node GPU Online: {out.strip()}")

    # Install iperf3 on Decode if needed and start daemon
    print("Ensuring iperf3 daemon is active on Decode Node...")
    run_ssh(decode_pub_ip, decode_direct_port, "apt-get update && apt-get install -y iperf3 iputils-ping net-tools")
    run_ssh(decode_pub_ip, decode_direct_port, "pkill iperf3; iperf3 -s -D")

    # Prefill direct
    prefill_direct_port = prefill.get('machine_dir_ssh_port')
    prefill_pub_ip = prefill.get('public_ipaddr')

    # Step 1: TCP Handshake RTT
    print(f"\n[1/3] Measuring Internal Subnet TCP Handshake RTT (Prefill -> {decode_ip}:5201)...")
    rtt_script = f"""python3 -c '
import socket, time, statistics, json
samples = []
for _ in range(25):
    t0 = time.perf_counter()
    try:
        s = socket.create_connection(("{decode_ip}", 5201), timeout=2)
        t1 = time.perf_counter()
        s.close()
        samples.append((t1 - t0) * 1000)
    except Exception as e:
        pass
    time.sleep(0.05)
if samples:
    print(json.dumps({{"min": min(samples), "avg": statistics.mean(samples), "max": max(samples), "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0, "samples": samples}}))
else:
    print("FAILED")
'"""
    rc, rtt_out, err = run_ssh(prefill_pub_ip, prefill_direct_port, rtt_script)
    print("RTT output:", rtt_out.strip())

    # Step 2: iperf3 Throughput (Prefill -> Decode)
    print(f"\n[2/3] Measuring Forward Bandwidth (Prefill -> Decode)...")
    rc_fwd, iperf_fwd, err_fwd = run_ssh(prefill_pub_ip, prefill_direct_port, f"iperf3 -c {decode_ip} -t 5 -J")
    if rc_fwd == 0 and iperf_fwd.strip():
        try:
            fwd_json = json.loads(iperf_fwd)
            bps = fwd_json.get('end', {}).get('sum_sent', {}).get('bits_per_second', 0)
            print(f"Forward Bandwidth: {bps / 1e6:.2f} Mbps ({bps / 8e6:.2f} MB/s)")
        except Exception:
            print("Forward raw output:", iperf_fwd.strip()[:300])
    else:
        print("Forward iperf3 failed or timed out:", err_fwd.strip())
    
    # Step 3: iperf3 Reverse Throughput (Decode -> Prefill)
    print(f"\n[3/3] Measuring Reverse Bandwidth (Decode -> Prefill)...")
    rc_rev, iperf_rev, err_rev = run_ssh(prefill_pub_ip, prefill_direct_port, f"iperf3 -c {decode_ip} -R -t 5 -J")
    if rc_rev == 0 and iperf_rev.strip():
        try:
            rev_json = json.loads(iperf_rev)
            bps = rev_json.get('end', {}).get('sum_received', {}).get('bits_per_second', 0)
            print(f"Reverse Bandwidth: {bps / 1e6:.2f} Mbps ({bps / 8e6:.2f} MB/s)")
        except Exception:
            print("Reverse raw output:", iperf_rev.strip()[:300])
    else:
        print("Reverse iperf3 failed or timed out:", err_rev.strip())

    print("\nNetwork measurements complete.")

if __name__ == '__main__':
    main()
