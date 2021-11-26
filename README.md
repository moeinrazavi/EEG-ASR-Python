# EEG-ASR-Python
Clean ASR (Artifact Subspace Reconstruction) in Python\n
After installing “pymanopt” package, replace the file “trust_regions.py” in the folder “/Users/user/opt/anaconda3/lib/python3.8/site-packages/pymanopt/solvers/” \nwith “trust_regions.py” in available in the repository.
Replace the file “_tensorflow.py” in the folder “/Users/user/opt/anaconda3/lib/python3.8/site-packages/pymanopt/tools/autodiff/” with the one available in the repository.
Open “Main.py”
Set Low and High edges of the Bandpass filter (line 18)
Give the name of EEG, Response and Stimulus Streams from the XDF file (lines 30-32)
Set the Sampling Frequency of the Device (line 39)
Set the path to XDF file (line 46)
Bandpass filter the EEG data (line 87)
Epoch the Data (lines 93-110)
Clean Data Drifts (line 122)
Clean Data Motion Artifacts using Artifact Subspace Reconstruction (line 123)
Run “Main.py” from terminal
