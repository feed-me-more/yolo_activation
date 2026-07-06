import socket
import time
import numpy as np
import struct
import cv2
import os
import hashlib


def send_packet(sock_tx, sock_rx, dest_addr, dest_port, frame_id, payload):
    # Split the payload into chunks that fit within the UDP payload size limit
    
    CHUNK_SIZE = 57600
    DATA_HEADER_FMT = "!I I d"
    ACK_HEADER_FMT = "!I"
    ACK_HEADER_SIZE = struct.calcsize(ACK_HEADER_FMT)
    num_chunks = (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE
    peak_rate = 0.0
    data_sent = 0
    retr_sent = 0

    send_start_time = time.perf_counter()

    for idx in range(num_chunks):
        start = idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = payload[start:end]

        # Send the packet with header and chunk
        packet = struct.pack(DATA_HEADER_FMT, idx, frame_id, time.time()) + chunk
        sock_tx.sendto(b"D" + packet, (dest_addr, dest_port))
        data_sent += len(packet) + 1
    
    send_duration = time.perf_counter() - send_start_time
    peak_rate = max((data_sent / send_duration) / (1024 * 1024), peak_rate)

#     print(f"Data sent for frame {frame_id}: {data_sent} bytes in {(send_duration * 1000):.3f} ms at {data_sent / send_duration / (1024 * 1024):.2f} MBps")

    while True:
        try:
            ack_data, _ = sock_rx.recvfrom(1024)

            if ack_data[0:1] == b"A":
                num_missing = struct.unpack("!I", ack_data[1: 1+ACK_HEADER_SIZE])[0]
                missing_packet_ids = struct.unpack("!"+"I"*num_missing, ack_data[1+ACK_HEADER_SIZE: 1+ACK_HEADER_SIZE+4*num_missing])
                                
#                 if num_missing != 0:
#                     print(f"ACK received for frame {frame_id}. Missing packets: {missing_packet_ids}")
                
                if num_missing == 0:
#                     print(f"All packets for frame {frame_id} acknowledged by receiver.")
                    return data_sent, retr_sent, peak_rate

                else:
                    print(f"ACK received for frame {frame_id}. Missing packets: {missing_packet_ids}")
                    for idx in missing_packet_ids:
                        start = idx * CHUNK_SIZE
                        end = start + CHUNK_SIZE
                        chunk = payload[start:end]
                        packet = struct.pack(DATA_HEADER_FMT, idx, frame_id, time.time()) + chunk
                        sock_tx.sendto(b"D" + packet, (dest_addr, dest_port))
                        retr_sent += len(packet) + 1                    
                    continue

        except socket.timeout:
            print(f"Timeout waiting for ACK for frame {frame_id}. Resending whole frame...")
            for idx in range(num_chunks):
                start = idx * CHUNK_SIZE
                end = start + CHUNK_SIZE
                chunk = payload[start:end]
                packet = struct.pack(DATA_HEADER_FMT, idx, frame_id, time.time()) + chunk
                sock_tx.sendto(b"D" + packet, (dest_addr, dest_port))
                retr_sent += len(packet) + 1
            print("Sent all frames again")
            continue
        
        except Exception as e:
            print("Error is: ",e)
            print("Data is: ",ack_data)


UDP_MAX_PAYLOAD = 65507
PILOT_HEADER_FMT = "!d I"
PILOT_HEADER_SIZE = struct.calcsize(PILOT_HEADER_FMT)

LOCAL_RECV_PORT = 5005        # Receiving PORT
LOCAL_SEND_PORT = 5006        # Sending PORT
DEST_PORT = 5006			  # Receiver's PORT
dest_addr = "10.51.25.143"   # Receiver's IP address (laptop)

source = "720p_traffic.mp4"  # Video source (file path)

data_sent_total = 0
retr_sent_total = 0
peak_throughput_mbps = 0

cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open source: {source}")

frame_size_bytes = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 3)
print(f"Frame size: {frame_size_bytes} bytes")

# Receiving socket
sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rx.bind(("", LOCAL_RECV_PORT))
sock_rx.settimeout(1)

sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_tx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
sock_tx.bind(("", LOCAL_SEND_PORT))
sock_tx.settimeout(0.5)

start_time = time.perf_counter()

# -------------------------Send a pilot packet to the receiver to signal the start of streaming------------------------
# Retry loop for sending pilot packet until ACK received
while True:             
    # start_time = time.time()
   
    try:
        sock_tx.sendto(b"P" + struct.pack(PILOT_HEADER_FMT, time.time(), frame_size_bytes), (dest_addr, DEST_PORT))
        data_sent_total += PILOT_HEADER_SIZE + 1

    except Exception as e:
        print(f"  Failed to send pilot packet: {e}. Retrying in 0.5 second...")
        time.sleep(0.5)
        continue

    try:
        while True:                
            data, _ = sock_rx.recvfrom(1024)

            if data.startswith(b"R"):
                print("Pilot packet acknowledged by receiver.")
                break

        break  # Exit the retry loop if ACK received

    except socket.timeout:
        print("  Timeout waiting for pilot ACK. Retrying pilot packet...")
        continue
# -------------------------Pilot packet sent and acknowledged. Start streaming video frames----------------------------
# ---------------------------------------------Streaming video frames--------------------------------------------------
frame_id = 1

while True:
    ret, frame = cap.read()                             
    frame_start_time = time.perf_counter()

    if not ret:
        break
    
#     print(type(frame), frame.shape)
    serialized_frame = frame.tobytes()
    
#     print(f"Hash for frame: {frame_id} is {hashlib.sha256(serialized_frame).hexdigest()}")
    
    data_sent, retr_sent, peak_throughput = send_packet(
        sock_tx=sock_tx,
        sock_rx=sock_rx,
        dest_addr=dest_addr,
        dest_port=DEST_PORT,
        frame_id=frame_id, 
        payload=serialized_frame
    )
    frame_id += 1
    
    data_sent_total += data_sent
    retr_sent_total += retr_sent
    peak_throughput_mbps = max(peak_throughput, peak_throughput_mbps)
    
    
actual_data = frame_size_bytes * (frame_id - 1)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# Send an end-of-stream packet to signal the receiver that streaming is complete
while True:                
    try:
        sock_tx.sendto(b"E", (dest_addr, DEST_PORT))
        data_sent_total += 1

    except Exception as e:
        print(f"  Failed to send end-of-stream  packet: {e}. Retrying in 0.5 second...")
        time.sleep(0.5)
        continue

    try:
        while True:                
            data, _ = sock_rx.recvfrom(1024)

            if data.startswith(b"F"):
                print("End-of-stream  packet acknowledged by receiver.")
                break

        break  # Exit the retry loop if ACK received

    except socket.timeout:
        print("  Timeout waiting for end-of-stream  ACK. Retrying...")
        continue

total_time = time.perf_counter() - start_time

print(f"Actual data: {(actual_data)/1024/1024} MB")
print(f"Total data sent: {(data_sent_total+retr_sent_total)/1024/1024} MB")
print(f"Total retransmission sent: {retr_sent_total/1024/1024} MB")
print(f"Total time taken: {total_time} seconds for {frame_id - 1} frames")
print(f"Average throughput: {(data_sent_total+retr_sent_total) / total_time / (1024 * 1024):.2f} MBps")
print(f"Peak throughput: {peak_throughput_mbps:.3f} MBps")
print(f"Total overhead: {(data_sent_total+retr_sent_total-actual_data)/actual_data * 100 :.4f} %")

cap.release()
cv2.destroyAllWindows()
