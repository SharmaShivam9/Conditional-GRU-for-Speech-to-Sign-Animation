**DES646: AI/ML for Designers** 

**Speech-to-Sign Language Animation Using AI-Driven Pose Learning Group Name: CBA** 

**Team Details:**

**Team Members:** Shivam Sharma (221016), Taneshwar Kumar Meena (221124) and Aryan Satyaprakash (220228)  ![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.002.png)

**Project Aim** 

This project aims to develop an **AI-based system capable of translating spoken sentences into sign-language animations** by learning the relationship between linguistic features and corresponding human pose data. The system seeks to improve accessibility for the hearing-impaired by providing real-time sign-language translations from natural speech. 

**Objectives** 

- **Data Processing and Representation:** Extract and normalize 3D pose coordinates (hands, face, and body) for each frame of sign-language videos using *Mediapipe*. 
- **Feature Learning:** Generate semantic embeddings for corresponding sentences and learn their correlation with pose data. 
- **Model Development:** Train a neural model to predict continuous pose sequences from textual embeddings. 
- **Speech-to-Text Integration:** Incorporate a live speech recognition pipeline to convert spoken input into text in real- time. 
- **Prototype Development:** Build a GUI that integrates speech recognition, text embedding, and pose animation for live demonstrations. 

**3. Methodology and Technical Approach** 

**Data Collection and Preprocessing**

The project utilizes data from the **iSign benchmark**, a critical resource for **Indian Sign Language (ISL) processing**. The corpus consists of video segments of continuous signing paired with English sentence translations, making it ideal for sign language translation tasks. 

**Initial Cleaning and Normalization:** Data preparation began with **127,237 sentences**. Initial cleaning involved: 

- Substituting all ampersand characters (&) with "and" (**2,120 sentences** affected). 
- A universal routine converting text to lowercase, replacing punctuation with spaces, and normalizing spaces to generate a consistent cleaned-text column. 

**Structural and Quality Filtering:** The normalized data was filtered to ensure quality. Structural constraints on character length (4 to 120) and word count (3) removed **10,239 sentences**. **Exact duplicates** were removed (**2,348 sentences**). A strict filter ensured sentences contained only lowercase letters and spaces, removing **25,484 sentences**. A final check removed **32 sentences** with high word repetition (ratio > 0.7). These steps removed **38,103 sentences**, leaving **89,134** high-quality sentences. 

**Grammar Check and Validation:** The **89,134 sentences** underwent two crucial validation steps: 

1. **Grammar Check:** An on-device LLM (Gemma-3.1b-it) filtered for grammatical correctness, retaining **27,305 sentences**. 
2. **Media Validation:** Cross-referencing against the video repository removed **239 sentences** with missing videos, resulting in **27,066** final validated video-text pairs. 

**Prototype Dataset Augmentation:** A final prototype dataset was created for training. A **Core Vocabulary** of **250 words** was defined from the corpus. A high-quality **'seed' dataset** was isolated, containing **417 sentences** that exclusively used the core vocabulary and met a minimum word frequency threshold. The seed was used to **augment the dataset** by generating **3,000 new examples** through weighted sentence recombination. 

**Sentence embedding:** We used the Universal Sentence Encoder (USE) model to create the sentence embeddings. For each sentence, it generates a 512-dimensional embedding using USE. The embedding is then saved as a separate .npy file named after its UID.  

**Final Dataset Summary:** The complete data pipeline yielded a total final dataset size of **3,417 embedded/vectorized sentences**. 

- **Total Overall Data Size:** **3,417 sentences**. 
- **Final Vocabulary Size:** **176 words** (unique words excluding stop words). 
- **Word Frequency Range:** The most common word ('know') appeared 453 times, and the least common words ('could', 'area') appeared 34 times.** 

**Pose File Generation and Normalization:** 

**The Pose Data File Structure:** The core input is a **Pose Data File** ($\text{.npy}$), generated from video footage using a pose estimation library like **MediaPipe**. Each file contains a NumPy array where every row represents a single video frame. The data within each frame comprises the concatenated x, y, and **z coordinates (3D data)**, along with a visibility score, for two sets of landmarks: 

- **Pose Landmarks:** 33 points covering the entire body (e.g., nose, shoulders, elbows, hips). 
- **Hand Landmarks:** 42 points (21 per hand) for the left and right-hand joints and fingertips. 

This 3D structure allows for the detailed capture of motion, which is critical for sign language analysis. 

**Smoothing, Scaling, and Dimensionality Reduction:** To ensure data stability, comparability across signers, and training efficiency, the following steps are applied: 

- **Data Smoothing:** Coordinates are passed through a **Savitzky-Golay filter** to smooth the signal via local polynomial regression. This reduces noise and jitter, ensuring the reconstructed movement is fluid. 
- **Scale Normalization (Using 3D Data):** The critical step is **scale normalization**. To make all signers appear the same size, a stable **scaling factor** is determined by comparing the median of the signer's true 3D shoulder width (calculated using x, y, z) against a fixed TARGET\_SHOULDER\_WIDTH. Using the z-coordinate is vital here to prevent perspective foreshortening from distorting the scale factor. All landmarks are multiplied by this factor, removing the influence of a signer’s size or distance from the camera. 
- **Centring and Standardization:** The landmarks are translated so the **midpoint between the shoulders** becomes the origin, centring the signing action. The torso is simplified by fixing the spine to a **CONSTANT\_SPINE\_LENGTH**, which focuses the analysis solely on the expressive motion of the arms and hands. 
- **Dimensionality Reduction (Final Output):** Since the machine learning model is trained on the 2D video projection, the z-coordinate and visibility scores are **discarded** after normalization is complete. The final pose file is saved with only the **normalized x and y coordinates** per landmark. This significantly improves data efficiency and directly aligns the input features with the downstream training task. Plus then we remove the useless landmark points and these are the final landmarks saved: 
- **Pose Landmarks:** 4 points covering only shoulders and elbows. 
- **Hand Landmarks:** 42 points (21 per hand) for the left and right-hand joints and fingertips. 
- So, total 46 2-D landmarks, making the shape of our pose.npy file as (92,T), where T is the total number of frames. 
- **Dimension Fixation and Masking:** We need a constant shape of pose files for our model to learn properly. So, we find Lmax, and for all the frames n, where T<n<Lmax, we the fake frames with the same coordinates as the last real frame. Then, we add a mask layer which contains 1 for real frames and 0 for fake frames.** 

**GIF Making** 

This file provides a crucial visualization tool for the project. Its sole purpose is to convert the raw numerical output of the PoseSeqModel (a sequence of 92-dimension vectors) into a human-readable, animated stick-figure GIF. 

**Key Function: generate\_stick\_figure\_gif\_from\_array** 

This function handles the entire conversion process from the model's prediction to a final .gif file. **Input:** 

- Y\_pred: A NumPy array with the shape (T, 92), where T is the number of frames and 92 is the pose vector. 
- output\_path: The file path to save the resulting GIF. 
- fps: The desired frames per second for the animation. 

**Visualization Process** 

The function iterates through the pose sequence and draws each frame onto a canvas. 

1. **Sub-sampling:** To create a smoother and faster-playing GIF, the function does not render every single frame. Instead, it processes every 3rd frame of the input sequence (Y\_pred[::3]). 
1. **Canvas Setup:** For each frame, it creates a blank, white 512x512 pixel image using numpy and cv2. 
1. **Decoding the 92-dim Vector:** The most critical step is decoding the flat 92-dimension vector for each frame. The script "knows" the structure of this vector: 
- **Body Pose (Features 0-7):** Reshaped into (4, 2) coordinates (left/right shoulder, left/right elbow). 
- **Left Hand (Features 8-49):** Reshaped into (21, 2) coordinates. 
- **Right Hand (Features 50-91):** Reshaped into (21, 2) coordinates. 
4. **Drawing the Skeleton:** 
- **Coordinate Scaling:** The normalized [0, 1] coordinates are scaled to the 512x512 pixel canvas. 
- **Head:** A head is drawn as a fixed-size circle, positioned relative to the midpoint of the shoulders. 
- **Body & Hands:** The script uses cv2.line to draw connections between the decoded (x, y) points based on pre-defined connection maps (SIMPLE\_POSE\_CONNECTIONS and HAND\_CONNECTIONS), rendering the arms, shoulders, and full hand skeletons. 
5. **GIF Compilation:** 
- Each rendered cv2 image (a numpy array) is appended to a list. 
- Finally, the imageio.mimsave function is used to compile this list of images into a single .gif file at the specified output\_path. 

**Pose Generation Model** 

The pose generator is a **sequence-to-sequence neural network** that learns to map linguistic embeddings to time-varying 3D pose sequences. 

**Model Architecture** 

The core of the project is the PoseSeqModel, a **conditional autoregressive sequence-to-sequence (seq2seq) model** implemented in PoseSeqModel.py. Its primary function is to map a single, static sentence embedding (a 512-dimension vector) to a dynamic, temporal sequence of poses (a series of 92-dimension vectors). 

The architecture is built on an **Encoder-Decoder** framework, but with a specific, non-standard implementation: 

- The **Encoder** is a simple feed-forward layer, not a recurrent network. Its sole purpose is to transform the input sentence embedding into the initial hidden state for the decoder. 
- The **Decoder** is a recurrent (GRU) network that generates the pose sequence one frame at a time. It does *not* use an attention mechanism. 

**1. Core Components** 

The model is constructed from four main components, defined in its \_\_init\_\_ method: 

1. **Encoder (Initial State Generator):** 
- This consists of a single **nn.Linear** layer followed by a **nn.ReLU** activation. 
- It maps the input sentence embedding (shape [Batch\_Size, sentence\_embed\_dim]) to a new vector with shape [Batch\_Size, n\_gru\_layers \* hidden\_dim]. 
- This vector is then reshaped to [n\_gru\_layers, Batch\_Size, hidden\_dim] to serve as the **initial hidden state (h0)** for the decoder's GRU. It effectively "primes" the decoder with the semantic meaning of the entire sentence. 
2. **Decoder (Conditional GRU):** 
- This is the generative core of the model, built from a multi-layer **nn.GRU** cell. 
- **No attention mechanism is used.** Instead, the model employs a simpler and highly effective method for maintaining context: the GRU's input\_size is defined as pose\_dim + sentence\_embed\_dim. 
- At *every time step* in the generation loop, the input to the GRU is a **concatenation** of two vectors: 
  - The pose from the *previous* time step (shape [Batch\_Size, 92]). 
  - The original, static sentence\_embedding (shape [Batch\_Size, 512]). 
- This "Conditional GRU" design continuously reminds the decoder of the global text prompt at every single frame it generates. This prevents the model from "forgetting" the prompt or diverging, which is a common problem in long sequence generation. 
3. **Start-of-Sequence (SOS) Token:** 
- This is a learnable nn.Parameter with the shape of a single pose frame ([1, 1, pose\_dim], or [1, 1, 92]). 
- This token acts as the very first input to the decoder (at $t=0$), serving as a "go" signal to kickstart the autoregressive generation process. 
4. **Output Layer:** 
- At each time step, the GRU's output (shape [Batch\_Size, 1, hidden\_dim]) is passed through a final nn.Linear projection layer. 
- This layer maps the high-dimensional hidden state down to the final pose dimension ([Batch\_Size, 1, 92]). 
- A **nn.Sigmoid** activation function is applied to this final output. This is a critical architectural choice, as it squashes all 92 output features to be strictly between 0 and 1. This matches the data normalization (as seen in make\_gif.py) where pose coordinates are scaled to a [0, 1] range. 

**2. Dual-Path Operation (The forward Method)** 

The model's forward method is built around a single for loop that unrolls for Lmax (575) time steps. This loop intelligently handles two distinct operational modes: 

1. **Inference Path (Autoregressive Generation):** 
- **Trigger:** Called when target\_sequence=None (as seen in prediction.py and GUI\_Test.py). 
- **Logic:** The model is purely **autoregressive**. The decoder\_input for the *next* step (t+1) is *always* the model's own (detached) output\_pose from the *current* step (t). 
- **Flow:** The model generates the entire sequence by "feeding itself" its own predictions, starting from the sos\_token. 
2. **Training Path (Scheduled Teacher Forcing):** 
- **Trigger:** Called when a target\_sequence (ground-truth) is provided (as seen in Final Training.py). 
- **Logic:** This mode uses **Scheduled Teacher Forcing** to stabilize training. At each step t, a "dice roll" (using random.random() < teacher\_forcing\_ratio) determines the input for step t+1: 
- **Teacher Forcing:** The model uses the **ground-truth pose** from the target\_sequence. This provides a stable learning signal and prevents the model from diverging. 
- **Student Forcing:** The model uses its **own predicted pose** (detached). This forces the model to learn how to recover from its own small errors, which is vital for good generation quality at inference time. 
- The teacher\_forcing\_ratio is decayed over epochs (as calculated in Final Training.py), gradually "weaning" the model off the teacher-forced data and preparing it for the inference-time reality. 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.003.png)

Fig. 1: Main model Architecture 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.004.png)

Fig. 2: Decoder Architecture 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.005.png)

Fig. 3: Training vs. Inference (Scheduled Teacher Forcing) **Loss Functions** 

The model's training is guided by a sophisticated composite loss function, defined in Loss\_func.py. This is not a single loss but a collection of multiple objectives, designed to balance pose accuracy with physical and temporal realism. 

The total objective combines several distinct components. Below is a detailed breakdown of each. 

1. **Pose L2 Loss (Lpose):** 

This is the primary supervisory loss, responsible for driving the model's accuracy. 

- **Purpose:** To directly minimize the error between the predicted pose and the ground-truth pose on a per-frame, per- joint basis. 
- **Mechanism:** It calculates the standard L2 (Euclidean) distance, or Mean Squared Error (MSE), between the prediction and target tensors. 
- **Masking:** This loss is critically important. It is only applied to *valid* frames. The mask tensor (shape [Batch, Lmax, 1]) is used to nullify the loss from any padded frames, ensuring the model is not penalized for its output after the true sequence has ended. 

**Mathematical Formula** Let: 

- B be the batch size. 
- T be the max sequence length (Lmax). 
- D be the pose dimension (92). 
- Ypred be the predicted pose tensor (shape [B, T, D]). 
- Ytarget be the ground-truth pose tensor (shape [B, T, D]). 
- M be the binary mask tensor (shape [B, T, 1]), where Mb,t = 1 for valid frames and 0 for padded frames. 
- Nvalid= sum (M) be the total number of valid feature elements in the batch. 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.006.png)

2. **Bone Consistency Loss (Lbone)** 

This is a complex, two-part regularization loss that enforces anatomical plausibility. It operates on the *predicted* pose, irrespective of the target pose. 

Before the loss can be calculated, the 31 bone lengths must be extracted from the 92-dimensional pose vector. **2a. Bone Pairs Definition** 

The file Loss\_func.py defines a function get\_bone\_pairs\_31() that creates a static list of 31 tuples. Each tuple (a, b) represents a "bone" by linking the index of its two endpoint joints. 

- **Structure:** The 31 bones are explicitly defined based on the 46-landmark skeleton (4 body + 21 left hand + 21 right hand). The list comprises: 
- **5 Upper Body Bones:** (e.g., (left\_shoulder, right\_shoulder), (left\_shoulder, left\_elbow)) 
- **13 Left Hand Bones:** (Connecting the 21 left hand landmarks) 
- **13 Right Hand Bones:** (Connecting the 21 right hand landmarks) 
- **Mechanism:** When the loss function is initialized, two tensors are created: \_idx\_a (a list of all "a" joint indices) and \_idx\_b (a list of all "b" joint indices). 
- **Usage:** In the bone\_consistency\_loss function, these index tensors are used to efficiently "gather" the (x, y) coordinates for all 31 pairs of joints from the Y\_pred tensor. This creates two new tensors, pa (all "a" points) and pb (all "b" points). 
- The Euclidean distance is then calculated between pa and pb to get a tensor la (shape [Batch, Lmax, 31]), which represents the **calculated length of all 31 bones for every frame in the batch.** 

This la tensor is the input to the two-part loss function described below. **2b. Part A: Bone Range Loss (Lrange)** 

- **Purpose:** To keep the length of each of the 31 defined bones within a pre-defined, physically realistic minimum (bone\_min) and maximum (bone\_max) range. 
- **Mechanism:** It calculates the length of all 31 bones for every valid frame. It then quadratically penalizes any length that falls outside the allowed bounds. 

**Formula:** Let lb,t,k be the calculated length of the k-th bone (out of 31) in batch b at time t. Let mink and maxk be the allowed min/max lengths for bone k. 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.007.png)

**2b. Part B: Bone Temporal Loss (Ltemp)** 

- **Purpose:** To ensure that bone lengths change *smoothly* over time. This prevents a "jittery" or "vibrating" appearance where bones rapidly change length between adjacent frames. 
- **Mechanism:** It calculates the squared difference of each bone's length between the current frame ($t$) and the previous frame (t-1). 

**Formula:** 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.008.png)

3. **Frame Length Loss (Llen)** 

This is a regularization loss that addresses how the model behaves in the *padded* portion of the sequence. 

- **Purpose:** In Final Training.py, target sequences are padded by *replicating the last valid frame*. This loss encourages the model to "stop moving" (i.e., have zero velocity and acceleration) in this padded region, matching the static nature of the padded target. 
- **Mechanism:** This loss is the *inverse* of Lpose. It *only* applies to invalid, padded frames (where M=0). It calculates the magnitude of the predicted velocity and acceleration in these regions and penalizes any non-zero values. 

**Formula:** Let M' = (1 - M) be the *invalid* mask. Let Vpredt = Ypredt - Ypredt-1 be the predicted velocity. Let Apredt = Ypredt - 2Ypredt-1 + Ypredt-2 be the predicted acceleration. 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.009.png) ![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.010.png)

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.011.png)

4. **Velocity Loss (Lvel)** 

This is a supervisory loss that operates on the *motion* of the valid frames, not just their position. 

- **Purpose:** To directly minimize the error between the *predicted velocity* (frame-to-frame change) and the *target velocity* of the ground-truth data. This encourages the model to match the *timing and speed* of the motion, not just the keyframes. 
- **Mechanism:** It calculates the velocity for both prediction and target (e.g., Vpredt = Ypredt - Ypredt-1). It then computes a weighted L2 loss on the difference. The weights (w\_vel) are pre-computed constants that apply different importance to the velocities of different joints. 

**Formula:** Let Vpred and Vtarget be the predicted and target velocities. Let Wv be the pre-defined weight tensor (wvel) for velocity. Let Mv be the mask for valid velocity frames (i.e., M from t=2 onwards). 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.012.png)

5. **Acceleration Loss (Lacc)** 

This loss is a higher-order version of the velocity loss, focusing on the *change in velocity*. 

- **Purpose:** To minimize the error between the *predicted acceleration* and the *target acceleration*. This helps the model learn smooth and realistic easing (speeding up and slowing down) of its movements, matching the ground truth. 
- **Mechanism:** It calculates the acceleration for both prediction and target (e.g., Apredt = Ypredt - 2Ypredt-1 + Ypredt-2). It then computes a weighted L2 loss on the difference, using a separate set of pre-computed weights (wacc). 

**Formula:** Let Apred and Atarget be the predicted and target accelerations. Let Wa be the pre-defined weight tensor (wacc) for acceleration. Let Ma be the mask for valid acceleration frames (i.e., M from t=3 onwards). 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.013.png)

6. **Derivation of Loss-Function Constants** 

The Loss\_func.py file contains several hard-coded tensors: bone\_min, bone\_max, w\_vel, and w\_acc. These values were not chosen arbitrarily but were pre-calculated by performing a statistical analysis of the entire training dataset. This analysis aims to ground the loss functions in the physical properties of the source data. 

**6a. Bone Min/Max Values** 

The bone\_min and bone\_max tensors provide the physical bounds for the Bone Consistency Loss. They are derived as follows: 

1. **Iterate Dataset:** The analysis script iterates through every pose sequence (arr) in the training data (pose\_data). 
1. **Calculate Bone Lengths:** For each sequence, it calculates the length of all 31 bones (as defined in get\_named\_bones\_31()) for *every frame*. 
1. **Find Sequence Min/Max:** It finds the minimum and maximum observed length for each bone *within that single sequence*. 
1. **Average Min/Max:** After processing all sequences, it computes the **average** of these minimums (avg\_min) and the **average** of these maximums (avg\_max) across the entire dataset. 

The resulting avg\_min and avg\_max vectors are the tensors stored as bone\_min and bone\_max in the loss file. They represent the average observed "natural" range of motion for each bone. 

**6b. Velocity & Acceleration Weights (w\_vel, w\_acc)** 

The w\_vel and w\_acc tensors provide per-joint (technically, per-coordinate) weights for the Velocity Loss and Acceleration Loss, respectively. They are calculated based on an "inverse motion energy" principle: 

1. **Calculate Motion:** The analysis script iterates through the dataset and calculates the per-joint velocity (and acceleration) for every frame. 
1. **Calculate Statistics:** It computes the dataset-wide mean (vmean) and standard deviation (vstd) of the motion for each joint. 
1. **Compute Motion Energy:** A "motion energy" metric is defined as: Emotion = mean\_motion + std\_motion. This value is high for joints that move a lot and have high variance (e.g., fingertips) and low for joints that are relatively stable (e.g., shoulders). 
1. **Compute Inverse Weight:** The weight is calculated as the inverse of this energy: W = 1 /( Emotion + epsilon) (where epsilon is a small value to prevent division by zero). 
1. **Normalize:** The final weights are normalized by dividing by their mean, so the average weight is 1.0. 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.014.png)![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.015.png)

**Implication:** This weighting scheme forces the model to be more precise with anatomically stable joints. Joints with *low* motion energy (like shoulders) get a *high* weight, meaning the loss function heavily penalizes any error in their velocity or acceleration. Conversely, joints with *high* motion energy (like fingertips) get a *low* weight, allowing the model more leniency. This focuses the model's effort on maintaining a stable core structure. 

**Voice Activity and Speech-to-Text Module** 

This supplementary module performs real-time Voice Activity Detection (VAD) and Speech-to-Text (STT) transcription for user input. 

1. VAD and Recording: The module first calibrates the ambient RMS noise level to set a recording threshold (1.6 times ambient RMS). It then initiates recording, which automatically stops when a period of silence exceeding 2.0 seconds is detected, saving the audio as a .wav file. 
1. Transcription: The .wav file is submitted to the Google Web Speech API for transcription. The resulting text is printed, and the temporary audio file is deleted, ensuring a clean process flow. 

**GUI and Real-Time Demonstration** 

The Speech-to-Sign Language (STS) Animator utilizes a modern, single-page Graphical User Interface (GUI) designed for clarity, responsiveness, and ease of use. The aesthetic is built upon **Glassmorphism**, featuring semi-transparent cards, vibrant glass borders, and subtle 3D lighting effects (powered by a custom Three.js background) to enhance depth and focus attention on the content panels. 

The main interface is split into a **2-column grid layout** (with a 1.2fr input column and a 1.3fr output column) which ensures a balanced visual hierarchy, dedicating appropriate space for both input controls and the resulting sign animation. 

1. **Left Panel: Input Source and Status (The Control Center):** The left column is divided vertically into two equal-sized cards, ensuring precise height alignment with the main animation output panel on the right. 
1. **Input Source Card:** This section allows the user to initiate the transcription and animation process. All interactive elements use bold, gradient-filled buttons to clearly indicate primary actions. 
- **Audio File (.wav, .mp3):** Primary button to upload a pre-recorded audio file. 
- **Start Live Audio Recording:** The central function, highlighted in a distinct **Danger/Red** color to emphasize its real-time, active status. 
2. **Transcription Output Card:** Displays the raw text transcribed from the user's audio input. It includes line spacing (line-height: 1.6) for optimal readability, allowing users to verify the transcription accuracy before viewing the final animation. 
2. **Right Panel: Animation Output (The Core Result):** This large, unified panel occupies the right column and serves as the primary output display. It is visually accented with an indigo border and a subtle glow to draw the user's eye to the main result. 
1. **Animation Container (GIF Block):** This flex-container area is where the generated sign language animation frames (simulated by images/GIFs) are displayed. 
- **Dynamic Flow:** The container uses flex-wrap and overflow-y: auto to manage multiple sign frames responsively, accommodating both short and long sentences. 
- **Sign Frame Styling:** Each individual sign frame utilizes a striking **gold-yellow border** and shadow, giving the output a polished, digital-art feel and making each translated word visually distinct.  
2. **Animation Control Footer:** This strip of control buttons is fixed to the bottom-right corner of the panel, ensuring essential controls are always accessible. 
- **Alignment:** All buttons are consistently aligned to the right (justify-content: flex-end). 
- **Save Generated GIF:** (Green/Success) Allows the user to download the final, complete animation. 
- **Reset/Clear:** (Yellow/Warning) Clears all text and output signs, returning the UI to its initial state. 
- **Back to Intro:** (Blue/Info) Provides a smooth transition back to the main introductory screen. 

![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.016.jpeg)![](Aspose.Words.45ce0e4b-eb3d-40ea-b0e6-2576a8cfbb55.017.jpeg)

**Results and Analysis** 

This section presents the qualitative and quantitative outcomes of the model's training and execution. The results demonstrate significant successes in learning anatomical structure and motion stability, but also highlight key limitations in dynamic motion generation, largely stemming from hardware and data-processing constraints. 

**Key Successes** 

1. **Anatomical & Structural Plausibility:** The model demonstrated a clear understanding of the human skeleton's structure. As seen in the generated GIFs, the model consistently produces well-formed hands and a stable torso. The intricate connections of the 21 hand landmarks are rendered correctly, and the bone lengths remain plausible. This confirms the bone\_consistency\_loss was highly effective in teaching the model a realistic anatomical prior. 
1. **Motion Stability and Smoothness:** The model's output is stable and free of the gross "jittering" often seen in generative-motion models. The visual results show a very fine-grained, subtle "vibration" or "wiggling" of the hands, but the limbs do not erratically jump or break. This indicates the velocity\_loss and acceleration\_loss were successful in regularizing the temporal dynamics, ensuring that the generated pose is at least stable, even if it is not dynamic. 

**Limitations and Model Failures** 

Despite the structural successes, the model failed to learn complex, human-like motion dynamics. The primary limitations are as follows: 

1. **"Average Motion" Failure:** The most significant failure, clearly visible in the output GIFs, is that the model does not generate *complete* motions. Instead, it appears to have learned the *average* pose associated with a prompt (e.g., "hands near chest"). The typical output settles into this static pose for the entire sequence, with only the previously mentioned, low-amplitude "wiggling" of the end-effectors (fingertips). It fails to produce continuous, large-scale dynamic actions like raising an arm, waving, or performing a gesture. 
1. **Hardware and Training Strategy Constraints:** 
   1. **Compute Limitations:** Training was heavily constrained by the use of consumer-grade GPUs and limited access to cloud platforms (Kaggle, Colab). 
   1. **Abandoned Two-Level Model:** The original plan involved a more complex, two-stage training strategy that was infeasible given the hardware limitations. This strategy would have first trained an encoder-decoder model *only* on pose data to learn a robust representation of human-like motion. The encoder would then be discarded, and the pre-trained decoder would be paired with a new text-encoder to learn the text-to-pose task. This approach was deemed to require months of training, which was not possible. 
1. **Data Pre-processing Limitations:** 
- **Text (Vocabulary):** The decision was made to use whole-sentence vectorization. While computationally efficient, this approach provides a highly limited vocabulary and fails to capture the nuance of complex sentences. A more robust method, such as per-word vectorization with stop-word removal and padding, was considered. However, this approach would have generated a dataset of unmanageable size for the available hardware. 
- **Pose (Masking):** The current implementation uses a single, simple mask for the pose sequences. A more advanced system was hypothesized, which would use three distinct masks (e.g., for loss calculation, encoder input, and decoder input) with different properties (float vs. bool, mask inversion, stop tokens). This added complexity was not implemented but remains a key area for future improvement. 

**Reflection** 

This project was an ambitious attempt at text-to-pose generation. Its greatest success was validating our physics-based loss functions. The bone\_consistency\_loss ensured the model learned the complex 46-landmark skeletal structure, producing anatomically plausible poses, not just random coordinates. Furthermore, the velocity\_loss and acceleration\_loss successfully eliminated temporal "jittering," resulting in stable, smooth outputs. 

However, the project's primary failure was its inability to generate dynamic, long-term motion. The model "cheated" by learning to produce a static, *average pose* for a given prompt, rather than a full action. 

This failure stems from two key constraints: 

1. **Data Representation:** Using whole-sentence embeddings compressed the entire action into one vector, making it too difficult for the decoder to "un-pack" into a sequence. 
1. **Hardware Limitations:** Lack of GPU power prevented a more robust two-stage training plan (pose-first, then text- conditional). 

Future work should execute this original two-stage plan, using per-word vectorization to provide a richer input signal. 

**Libraries and Technologies used** 

The project leverages a combination of deep learning frameworks, data manipulation libraries, and visualization tools to build the complete text-to-pose pipeline. The technologies are used across different phases: text encoding, model training, and visual output generation. 

1. **Deep Learning & Machine Learning** 
- **PyTorch (torch)**: This is the primary deep learning framework used for the entire project. 
- **torch.nn**: Provides the building blocks for the PoseSeqModel, including the nn.Linear, nn.GRU, nn.ReLU, and nn.Sigmoid layers. 
- **torch.optim**: Used in Final Training.py to implement the Adam optimizer for training the model. 
- **torch.utils.data**: Used in Final Training.py to create the TensorDataset and DataLoader for efficient, batched training. 
- **TensorFlow Hub (tensorflow\_hub)**: This library is used for a single, critical task in GUI\_Test.py: downloading and running the pre-trained **Universal Sentence Encoder (USE)**. This model is responsible for converting the input text sentences into 512-dimensional vector embeddings, which are the starting input for the PoseSeqModel. 
2. **Data Handling & Numerical Computation** 
- **NumPy (numpy)**: The fundamental library for numerical computation. It is used in Final Training.py, prediction.py, and GUI\_Test.py to load and pre-process the .npy files containing the sentence embeddings and pose data. It is also used in make\_gif.py to create and manipulate the image canvas. 
3. **Visualization & Image Processing** 
   1. **OpenCV (cv2)**: Used in make\_gif.py as the primary drawing engine. It creates the blank white canvas for each frame and draws the lines and circles that constitute the stick-figure skeleton. 
   1. **ImageIO (imageio.v2)**: Used in make\_gif.py to compile the sequence of individual image frames (generated by OpenCV) into the final, animated .gif file. 
3. **Python Standard Library** 
   1. **os**: Used extensively across all scripts for path manipulation (os.path.join), creating directories (os.makedirs), and listing files (os.listdir) to find the training/testing data. 
   1. **random**: Used in Final Training.py to implement the stochastic "dice roll" for Scheduled Teacher Forcing and in prediction.py to select a random test sample. 
   1. **time**: Used in Final Training.py to time the duration of training epochs and other operations for logging. 
   1. **math**: Used in Final Training.py to calculate the sigmoid decay for the teacher forcing ratio. 
3. **GUI:** 
- Python (Flask backend, ML inference) 
- HTML 
- CSS 
- JavaScript 
- Three.js (WebGL) 
