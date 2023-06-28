# ArrowDetection

## STEPS
<br />
<b>Step 1: </b> Clone the repository
<br/><br/>
<b>Step 2: </b> Create a virtual environment
<pre>
python -m venv OD_1    #can use any name instead of 'OD_1'
</pre>
<br/>
<b>Step 3: </b> Activate the virtual environment
<pre>
source OD_1/bin/activate
</pre>
<br/>
<b>Step 4: </b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=OD_1    #use the same name as the venv created
</pre>
<b>Step 5: </b> Install required dependencies.<br/>
1. Install opencv
<pre>
pip install opencv-pyhton
<pre>
2. <p> Run the 'Installing_Dependencies.ipynb' file to install the Object Detection API on the 'OD_1' kernal. Some steps regarding how to identify if the Object Detection API is installed are mentioned within the Notebook.<br/>
<b>Step 6.1: </b> Running Image Detection
<p>Create a folder named 'test' and add images within it.<br> 
<p>Save this folder in Tensorflow/workspace/images<br>
<p>Then run ImageDetection.py
<b>Step 6.2: </b> Running Real Time Object Detection
<p> start running RealTimeDetection.py.<br>
<p> The video will start running in a few seconds and then you can test it out.<br/>  


# Arrow Detection Dataset and labels

https://drive.google.com/drive/folders/1Reb3V9Y__r4dPYrWwIzAYH-Y4QQ0ZJU6?usp=drive_link
