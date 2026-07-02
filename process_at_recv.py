import pickle
from multiprocessing import Process, Queue, Value
from queue import Empty
import socket
import struct
import time
import numpy as np
import cv2
from ultralytics import YOLO
from compare_accuracy import average, compare_detections
from temp_plot import plot_data_rate, plot_latency


frame_queue = Queue(maxsize=25)
main_start_time = Value('d', 0.0)  # Shared variable to store the start time of the main process
producer_finished = Value('b', False)


def receiver():

    try: 
        HEADER_FMT = "!I I d"
        HEADER_SIZE = struct.calcsize(HEADER_FMT)
        UDP_MAX_PAYLOAD = 65507
        CHUNK_SIZE = 57600

        LOCAL_RECV_PORT = 5006            # Receiving port
        LOCAL_SEND_PORT = 5005            # Sending port
        DEST_PORT = 5005                  # Receiver's port
        # dest_addr = "192.168.0.163"       # Receiver's IP address
        dest_addr = "10.51.1.167"
        # global main_start_time

        # Receiving socket
        sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        # print(f"  Binding to {LOCAL_RECV_PORT}...")
        sock_rx.bind(("", LOCAL_RECV_PORT))
        sock_rx.settimeout(0.5)

        sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_tx.bind(("", LOCAL_SEND_PORT))
        sock_tx.settimeout(0.5)

        ret_ack = 0
        latency_accumulator = {}
        data_rate = {}

        print("Waiting for pilot packet...")


        #-------------------------------------------------- Receive a pilot packet ----------------------------------------------

        # Retry loop for waiting for pilot packet and sending ACK
        while True:
            try:
                while True:                
                    data, sender_addr = sock_rx.recvfrom(1024)

                    if data.startswith(b"P"):
                        main_start_time.value, frame_size_bytes = struct.unpack("!d I", data[1:])
                        print(f"Pilot packet received. Start time: {main_start_time.value}, Frame size: {frame_size_bytes} bytes")

                        sock_tx.sendto(b"R", (sender_addr[0], DEST_PORT))               
                        break
                break

            except socket.timeout:
                print("  Timeout waiting for pilot")

        pilot_recv_time = time.time()
        print(f"Pilot received in {(pilot_recv_time - main_start_time.value)*1000:.3f} ms")
        #-------------------------------------------------- Pilot packet received ------------------------------------------
        #---------------------------------------------- Receiving loop for each frame---------------------------------------
        frame_id = 1
        total_pkts = (frame_size_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE  # Calculate total packets needed for the frame
        
        while True:

            exp_ids = set(range(total_pkts))  # Expected packet IDs for the current frame
            buffer = {}
            frame_recv_start = 0
            frame_recv_stop = 0
            
            while True:
                try:
                    data, sender_addr = sock_rx.recvfrom(UDP_MAX_PAYLOAD)

                    if data[0:1] == b"D":
                        idx, pkt_frame_id, recv_start = struct.unpack(HEADER_FMT, data[1:HEADER_SIZE+1])
                        # print(f"Received packet {idx} of frame {pkt_frame_id} frame id is {frame_id}")
                        
                        latency = (time.time() - recv_start) * 1e3
                        # print(f"Packet {idx} of frame {pkt_frame_id} received in {latency:.3f} ms")

                        latency_accumulator.setdefault(pkt_frame_id, []).append(latency)
                        # print(f"Latency for frame {pkt_frame_id}, packet {idx}: {latency:.3f} ms")

                        if frame_recv_start == 0:
                            frame_recv_start = recv_start

                        if pkt_frame_id != frame_id:
                            print(
                                f"WARNING: got packet for frame {pkt_frame_id} "
                                f"while assembling frame {frame_id}"
                            )
                            continue

                        buffer[idx] = data[HEADER_SIZE+1:]
                        
                        # print(f"Received packet {idx} of frame {frame_id} from {sender_addr[0]} with payload size {len(data[HEADER_SIZE+1:])} bytes")
                        exp_ids.discard(idx)

                        if not exp_ids:
                            print(f"All packets for frame {frame_id} received.")
                            missing_ids = sorted(exp_ids)
                            ack_data = b"A" + struct.pack("!I" + "I" * len(missing_ids), len(missing_ids), *missing_ids)
                            sock_tx.sendto(ack_data, (sender_addr[0], DEST_PORT))
                            frame_recv_stop = time.time()
                            break

                    if data[0:1] == b"E":
                        print(f"End of frame signaled by sender.")
                        sock_tx.sendto(b"F", (sender_addr[0], DEST_PORT))
                        frame_queue.put(None)  # Signal processor to stop
                        sock_tx.close()
                        sock_rx.close()
                        producer_finished.value = True
                        print(" Total retries for ACKs: ", ret_ack)
                        
                        plot_data_rate(data_rate, "Server_side_data_rate")

                        all_lats = np.array([
                        lat
                        for frame_lats in latency_accumulator.values()
                        for lat in frame_lats
                        if lat is not None])

                        print(f"Overall average packet latency: {np.mean(all_lats):.3f} ms")
                        print(f"Overall max packet latency    : {np.max(all_lats):.3f} ms")
                        print(f"Overall min packet latency    : {np.min(all_lats):.3f} ms")
                        print(f"Overall 95th percentile latency: {np.percentile(all_lats, 95):.3f} ms")

                        plot_latency(latency_accumulator, "Server_side_latency_scatter", np.percentile(all_lats, 95))

                        return

                except socket.timeout:
                    missing_ids = sorted(exp_ids)
                    ack_data = b"A" + struct.pack("!I" + "I" * len(missing_ids), len(missing_ids), *missing_ids)
                    sock_tx.sendto(ack_data, (sender_addr[0], DEST_PORT))
                    ret_ack += 1
                    print(f"Timeout: missing packets {missing_ids} for frame {frame_id}. Sent ACK with missing packet IDs.")
                    continue    
            
            frame_time = frame_recv_stop - frame_recv_start
            data_rate[frame_id] = frame_size_bytes / frame_time / (1024 * 1024)  # in MB/s
            print(f"Whole frame {frame_id} received in {frame_time*1000:.3f} ms at rate of {frame_size_bytes/frame_time/1024/1024:.3f} MB/s")
            
            frame_recv_start = 0

            serial_frame = b"".join(buffer[i] for i in range(total_pkts))
            serial_frame = serial_frame[:frame_size_bytes]  # Trim to expected frame size in case of extra bytes

            print(f"Reassembled frame {frame_id} from {len(serial_frame)} bytes of data")

            frame = np.frombuffer(serial_frame, dtype=np.uint8).reshape((360, 640, 3))  # Assuming 720p RGB frames

            # cv2.imshow("Received Frame", frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            frame_queue.put(frame)
            frame_id += 1

            print(f"Waiting for the frame {frame_id}...")

    except Exception as e:
        print(f"Receiver encountered an error: {e}")    

def processor():

    frame_id = 0
    # start_time = 0
    inference_time_ms = 0.0    
    max_inf_time = 0.0
    max_inf_frame = -1
    total_inf_time = 0.0
    result_metric = []
    frame = None

    model = YOLO("yolov8n.pt")
    VEHICLE_CLASSES = {1, 2, 3, 5, 7}
    try:
        # while frame_queue.empty():
        #     print("Processor waiting for frames...")
        #     time.sleep(0.5)

        with open("reference_det_set_640_360.pkl", "rb") as f:
            reference_dets = pickle.load(f)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            "processed_video.mp4",
            fourcc,
            30,
            (640, 360)
        )

        
        once = True
        while True:
            try:
                frame = frame_queue.get(timeout=4) 

                if frame is not None and once:
                    print("First frame received. Starting processing...")
                    process_start_time = time.perf_counter()
                    once = False

                frame_start_time = time.perf_counter()

                if frame is None:  # Check for stop signal
                    print("No more frames to process. Exiting processor.")
                    video.release()
                    break

                # Process the frame
                # frame = cv2.resize(frame, (640, 640))  
                results = model(frame, verbose=False)              
                annotated = frame.copy()
                recv_detections = []

                for box in results[0].boxes:
                    cls = int(box.cls.item())

                    if cls in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf.item())

                        recv_detections.append([x1, y1, x2, y2, conf, cls])

                        label = f"{model.names[cls]} {conf:.2f}"

                        cv2.rectangle(
                            annotated, 
                            (x1, y1), 
                            (x2, y2), 
                            # 0.7 * (0, 255, 0) + 0.3 * (255, 0, 0),
                            (255, 0, 0),
                            4,
                        )
                        # Class name and confidence
                        cv2.putText(
                            annotated,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

                reference_det = reference_dets.get(frame_id,np.zeros((0, 6), dtype=np.float32))
                recv_detection = np.array(recv_detections, dtype=np.float32) if recv_detections else np.zeros((0, 6), dtype=np.float32)

                frame_end_time = time.perf_counter()
                inference_time_ms = (frame_end_time - frame_start_time) * 1e3

                # Text on top-left corner with frame ID
                cv2.putText(
                    annotated,
                    f"frame {frame_id+1} in {inference_time_ms:.1f} ms",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # cv2.imshow("Vehicle Detection", annotated)

                if (inference_time_ms > max_inf_time) and frame_id > 0:
                    max_inf_time = inference_time_ms
                    max_inf_frame = frame_id + 1

                total_inf_time += inference_time_ms
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
                annotated = cv2.resize(annotated, (640, 360))
                video.write(annotated)

                metrics = compare_detections(recv_detection, reference_det)

                result_metric.append({"frame_id": frame_id + 1, **metrics})
                # print("Metric:", metrics)

                frame_id += 1

            except Empty:
                if producer_finished.value:
                    print("Processor timed out waiting for frames. Exiting.")
                    break

                print("Waiting for more frames...")
                time.sleep(0.2)
                continue

        average_metrics = average(result_metric)
        print(f"Average metrics over {len(result_metric)} frames: {average_metrics}")

        total_process_time = time.perf_counter() - process_start_time

        print(f"Processed {frame_id} frames in {total_process_time:.3f} s ") 
        print(f"Average inference time: {total_inf_time / (frame_id-1) :.3f} ms per frame")
        print(f"Max inference time: {max_inf_time:.1f} ms on frame {max_inf_frame}")

        total_main_time = time.time() - main_start_time.value
        print(f"Total time from pilot packet to end of processing: {total_main_time:.3f} seconds")
        return

    except Exception as e:
        print(f"Processor encountered an error: {e}")

def main():

    print("Starting receiver and processor processes...")

    try:
        receiver_proc = Process(target=receiver)
        processor_proc = Process(target=processor)

        receiver_proc.start()
        processor_proc.start()

        receiver_proc.join()
        processor_proc.join()
    
    finally:
        cv2.destroyAllWindows()
        receiver_proc.terminate()
        processor_proc.terminate()

        receiver_proc.join()
        processor_proc.join()


if __name__ == "__main__":
    print("Starting main process...")
    main()
