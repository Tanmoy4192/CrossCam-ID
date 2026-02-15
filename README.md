<h1>CrossCam ID</h1>

<p>
CrossCam ID is a person re-identification system that works with live cameras and uploaded videos.
It detects people, assigns consistent IDs, and shows when the same person appears again.
The system works both in real-time camera streams and through a web interface where users can upload videos for analysis.
</p>

<hr>

<h2>What this project does</h2>

<ul>
<li>Detects people in video frames.</li>
<li>Creates an embedding (feature vector) for each person.</li>
<li>Assigns an ID to each person and keeps it consistent across frames.</li>
<li>Tracks recurring appearances of the same person.</li>
<li>Works with multiple cameras running at the same time.</li>
<li>Allows users to upload videos and analyze appearances using a web interface.</li>
<li>Shows timestamps and preview images of detected persons.</li>
<li>Can compare two videos and detect if the same person appears in both.</li>
</ul>

<hr>

<h2>Why this can be useful</h2>

<p>
Reviewing surveillance footage manually is slow and repetitive.
This system helps by automatically showing:
</p>

<ul>
<li>When a person appeared in a video.</li>
<li>How many times they appeared.</li>
<li>Preview images of those appearances.</li>
<li>If the same person appears in two different videos.</li>
</ul>

<p>
This type of workflow can help in situations like security footage review, office monitoring, or event recordings where repeated appearances need to be located quickly.
</p>

<hr>

<h2>Project structure</h2>

<pre>
project/
│
├── main.py              # Real-time multi-camera system
├── detectPerson.py      # Person detection using YOLOv8
├── encode.py            # Embedding extraction using TorchReID
├── tracker.py           # ID assignment using cosine similarity
├── camera.py            # Camera/video source wrapper
│
├── web/
│   ├── app.py           # FastAPI backend
│   └── templates/
│       ├── index.html   # Web UI
│       └── style.css    # Styling
│
└── requirements.txt
</pre>

<hr>

<h2>Installation</h2>

<p>Clone repository:</p>

<pre>
git clone https://github.com/your_username/crosscam-id.git
cd crosscam-id
</pre>

<p>Create virtual environment (optional but recommended):</p>

<pre>
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
</pre>

<p>Install dependencies:</p>

<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>Running real-time multi-camera system</h2>

<p>Run:</p>

<pre>
python main.py
</pre>

<p>
Two camera windows will open.  
Press <b>q</b> to exit.
</p>

<p>
Camera source can be:
</p>

<ul>
<li>Webcam index (0, 1, etc.)</li>
<li>IP camera stream URL</li>
<li>Video file path</li>
</ul>

<hr>

<h2>Running web application</h2>

<p>Start FastAPI server:</p>

<pre>
uvicorn web.app:app --reload
</pre>

<p>Open browser:</p>

<pre>
http://localhost:8000
</pre>

<p>
Upload one or two videos and analyze appearances.
</p>

<hr>

<h2>Web analysis output</h2>

<ul>
<li>If one video is uploaded:
    <ul>
        <li>Shows people appearing more than once.</li>
        <li>Displays timestamps and preview images.</li>
    </ul>
</li>
<li>If two videos are uploaded:
    <ul>
        <li>Checks whether the same person appears in both.</li>
        <li>Displays appearance timestamps for each video.</li>
    </ul>
</li>
</ul>

<hr>

<h2>Notes</h2>

<ul>
<li>Processing speed depends on hardware.</li>
<li>GPU improves performance but CPU also works.</li>
<li>Large videos take longer to process.</li>
<li>System currently runs fully locally.</li>
</ul>

<hr>

<h2>Demo — Real-time System</h2>

<!-- Replace link or embed demo GIF/video -->
<p>Add demo video or GIF showing multi-camera real-time tracking here.</p>

<hr>

<h2>Demo — Web Video Analysis</h2>

<!-- Replace link or embed demo GIF/video -->
<p>Add demo video or GIF showing video upload and analysis here.</p>

<hr>

<h2>Future improvements (optional)</h2>

<ul>
<li>Database storage of identities.</li>
<li>Better performance optimization.</li>
<li>Web dashboard for statistics.</li>
<li>Support for more cameras.</li>
</ul>

<hr>

<h2>Author</h2>

<p>
Tanmoy Samanta
</p>
