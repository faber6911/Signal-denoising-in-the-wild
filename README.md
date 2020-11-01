<div align = "center"><p align="center">
  <a href="http://datascience.disco.unimib.it/it/"><img src = "https://raw.githubusercontent.com/malborroni/Foundations_of_Computer-Science/master/images/DSunimib.png" width = "100%"></a>
</p>
</div>

<h1 align = "center">Audio denoising in the wild</h1>
<h6 align = "center">a Master Degree Thesis project</h6>

<p align="center">
  <a href="#overview">Overview &nbsp;</a>|
  <a href="#data">Data &nbsp;</a>|
  <a href="#report">Report &nbsp;</a>|
  <a href="#presentation">Presentation &nbsp;</a>|
  <a href="#aboutme">About me</a>
</p>


<a name="overview"></a>
## &#9741; &nbsp; Overview
<p align="justify">Technological innovation and the large-scale application of highly innovative tools have been offering great opportunities for study and development in the field of Artificial Intelligence for years.
Among the numerous fields of application of these technologies are voice assistants with all the tasks associated with them.
The very nature of these technological solutions means that their use often takes place in hostile, or highly noisy, environments such as urban contexts.
From this problem arises the opportunity to investigate the potential of an end-to-end approach that includes a Deep Learning model to perform the denoising task in direct communication with a second model whose goal is to operate the speaker classification.
This thesis work aims to verify this potential through a structured path, organized in numerous phases whose purpose is to obtain timely and comparable measurements with previous and subsequent works.</p>

<a name="data"></a>
## &#9741; &nbsp; Data
The <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/data">data</a> folder contains all the scp (script) files with the paths leading to the audio files in my machines. 

<a name="code"></a>
## &#9741; &nbsp; Code
<p align="justify">The code is divided between the <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/pyscripts">pyscripts</a>, <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/SpeakerRecognition">SpeakerRecognition</a> and <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/plots">plots</a> folders. 
  
These respectively contain the code for the first training of the WaveNet models, the part of the code dedicated to the fine-tuning phase and the one dedicated to the creation of the visualizations. 

The file <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/blob/master/kalditorch.yml">kalditorch.yml</a> contains all the packages needed to reproduce the system.

The <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/log">log</a> folder contains alla the checkpoints created during the models training.</p>

<a name="report"></a>
## &#9741; &nbsp; Report
The text of the thesis in pdf format is available <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/report/audio_denoising_in_the_wild.pdf">here</a>.
Acknowledgments are yet to be defined.

<a name="presentation"></a>
## &#9741; &nbsp; Presentation
Still work in progress.
<!--The presentation slides of the thesis work are available <a href="https://github.com/faber6911/Signal-denoising-in-the-wild/tree/master/presentation/thesis_presentation.pdf">here</a>.-->

<a name="aboutme"></a>

## &#9741; &nbsp; About me

&#8860; &nbsp; **Fabrizio D'Intinosante**

- *Current studies*: MSc Data Science student at University of Milan-Bicocca;
- *Previous studies*: BSc in Economics and Statistics at University of Turin.
<br>

<p align = "center">
  <a href = "https://www.linkedin.com/in/fabrizio-d-intinosante-125042180/"><img src="https://raw.githubusercontent.com/DBertazioli/Interact/master/img/iconfinder_Popular_Social_Media-22_2329259.png" width = "3%"></a>
  <a href = "https://faber6911.github.io/"><img src="https://raw.githubusercontent.com/malborroni/Foundations_of_Computer-Science/master/images/GitHub.png" width = "3%"></a>
</p>
