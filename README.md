**send_video.py**

Sends the video to the receiver/server frame by frame using UDP based retransmission.
Sender/client sends a pilot packet marking the start of transmission and waits for its ACK. 
Once received, each frame is then serialized into bytes and divided into packets. Each packet is appended with a packet ID and sent to the receiver/server, where ACK is sent back if all packets are received or if the receiver times out and in each case the receiver sends the ACK containing the missing packet IDs. Sender then retransmits the missing packets if any or else moves to the next frame.


**process_at_recv.py**
Contains 2 functions receiver and processor. Both are implemented to run in multiprocessing fashion.

	**receiver**
	
	Receives the pilot packet to fetch the sender's timestamp and then receives each frame's packets in a loop.  Store packets corresponding to each frame IDs in a dictionary and trigger retransmission for any missing packet IDs. If all packets corresponding to the frame are received, the frame is pushed into a Queue.
	
	**processor**
	
	Fetches frame from the Queue and runs YOLO model for vehicle detection. After detection appends the results on the frame and writes it into a video file.


**streaming.sender.py**

Partially runs the model on each frame and send the activations to the server/receiver. 
Preprocess each frame: Convert BGR --> RGB --> PIL --> Normalizes float tensor in range [0, 1] 
Partially process the frame and form a flat single vector for the feature maps, divides into chunks and send.
Waits for the results and if valid result received, annotate and write it into video.


**streaming.receiver.py**

Receives the activations from the client/sender and process the remaining layers at the server.
Receives the frame packets, if all packets received in frame timeout time then process it normally or if all packets of a frame not received then mark it as a stale frame and reconstruct it by using the data from corpus training for vehicle detections.
Once the frame is processed, send the detection results along with other parameters back to client/sender.


**streaming.sender_noperm.py**

Same as sender.py but the data is interleaved based on a pattern randomly chosen from a fixed set of patterns in "interleave_patterns.npy" created using "generate_interleave_patterns.py". 
Interleaving helps in avoiding accuracy loss due to burst error.


**streaming.receiver_noperm.py**

Same as receiver.py except the incomplete packets are replaced by zeros instead of corpus data.


**vehicle_det.py**

Runs the model on the video and saves the detection results frame by frame. 
Compares the detection results with the reference detection results and calculate the metrics for accuracy as defined in "compare_accuracy.py".


**compare_accuracy.py**

Contains 2 functions 

	**compare_detection**

	Calculates the precision, recall, f1-score, mean-IoU and confidence difference of the detection results with respect to reference detection results.
	The metric corresponding to all the frames

	**average**

	Takes average over all the frames to yield a single metric values for the detection.


  
