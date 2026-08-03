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


  
