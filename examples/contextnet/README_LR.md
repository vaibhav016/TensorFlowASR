# Welcome to experiments on low rank approximation and understanding of feature representations in deep nets on speech. 
This project is started and guided by Dr Vinayak Abrol.(1 April 2021) 

###Some useful resources

1) Below are links of Research papers that we must read before proceeding further 
    1) http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf
        _this is sir's paper and new idea of low rank filtering which we need to implement._
    
    2) https://arxiv.org/pdf/2005.03191.pdf  _this is contextNet, over which we need to implement new ideas._
    3) Following accounts are used for data storing purposes. 
        1) experiments.asr@gmail.com
        2) experiments.asr2@gmail.com
        3) `For password, kindly ask sir`
        
    
2) Dataset used in this research work is Librispeech and the link to all the files is in tensorflow_asr/scripts/Datasets/download_links.sh
    1) Training comes with 3 types of files, clean-100(100 denotes hours of speech), clean-360, and other-500(unclean data)__
    2) Similarly testing is also available as test-clean and test-other(for more rigorous testing)__
    3) For validation dev-clean and dev-other are provided__
    4) Go through the above shell script to understand how data is unpacked and how transcripts are fed to the mode__
    5) Understand the config file, different configurations for model parameters, featurizers, learning parameters are systematically placed in the config file__
    
3) We have access to following machines - 
    1) quadro rtx 4000 VM (conda env- beta)
    2) v100 (conda env - tf2)
    3) a100 (conda env - tf2)
    
    __Kindly do not change or modify the conda environment. Downloading any package or updating should be done only when utmost necessary.__
    __As a rule of thumb, always create a seperate conda environment and perform experiments__ 
    __For convenience, on every machine, an environment.yml file is provided which can be used to download packages, drivers, etc. This would save a lot of time__
    
4) A general pipeline for this research work is as follows- 

    1)Specify the model paramenters and layers
    2) Train the model( either on whole, or clean, depending on the requirements and time constraints.
    3) From the trained models, generate the following metrics
        1) Get the accuracy metrics(word error rate)
        2) Generate loss landscapes
        3) Generate the integrated gradients landscapes.
        
5) Now lets understand how to setup this project- 

    1) log into the available machine. For illustration purposes, lets assume we have access to v100 machine.
    2) `cd ASR/TensorFlow`  Already this folder has all the working code along with git setup. If this is available, then we are good to go to step 4. If not then follow step 3
    3) make a folder ASR 
        1) mkdir ASR 
        2) cd ASR 
        3) git clone `https://github.com/vaibhav016/TensorFlowASR.git`
        4) cd TensorFlow
        5) git checkout saliency_visualisation(This branch has the most updated code and workflow)
   
    5) At the base folder(/Tensorflow), run the following two commands 
         1) `python setup.py build`
         2) `python setup.py install`
         3) Since this project uses setuptools, so any modification done in tensorflow_asr folder will require you to run these two commands at the base folder. Only then the changes would be reflected.
         
    6) conda create -n {env_name} -f {path to environment.yml}  __env file will be found in the drive link:  https://drive.google.com/file/d/1eDKDIn3Po4ZBL_xVymL7XPTt_M2jfA8h/view?usp=sharing__
    7) Now cd to examples/contextnet.
    8) This folder has all the necessary files to train, and visualise. 
    9) Have a look at tensorflow_asr folder. This folder has all the files related to featurisers and model declaration. Most of our time will go here, in modifying models(especially contextnet)
    
    10) To download data( v100 and a100 have data already the datasets so just move them into the desired directory)
    11) In case of downloading, just __cd into scripts/Datasets and run download.sh.__ It will automatically download and place the data according to the directory structure defined in the config files.
    12) Create transcripts by running __scripts/create_transcripts_from_data.sh__(You may modify the file according to your needs)
    13) This scritp calls the python file create_librispeech_trans.py, so study this file. If any error comes regarding missing package, then check tensorflow-io version. With tensorflow==2.5.0, tensorflow-io will have to be 0.18.0
   
    14) Run train.py for training.
    15) Run test.py for testing
    16 Store the test results in a methodical way on some excel sheet. 
    17) For visualisations, we have two kinds of scripts.
        1) for loss landscapes, cd into context_visualisation/loss_landscape_visualisation.
            1) run generate_lists.py(This generates the loss and accuracy lists)
            2) now run plot_loss.py (From those lists, images are drawn both 2d and 3d)
            3) now run video_create.py(It sews all the images into a single video)
        
        2) For gradient visualisation, 
            1) run integrated_grad_vis.py, which will generate the integrated gradients for all the trained models
            2) then run plot_gradients.py
            3) Finally run video_create.py
            
    **WORD OF CAUTION**- whenever any change is made to the code and scripts are run, then you would need to run the above two commands first, because setuptools install
    the packages in env/lib/site-packages/python3.8/XXXX. So your change will not be reflected unless you build and install again. 

Since we need logging into files and in background to free terminal so better run this command
     `_nohup python3 train.py > logging.out &_`
     
if you want error in a seperate file then 
         `_nohup python3 train.py > logging.out 2>&1 &_`
 
**make sure to download and upload trained models to drive and keep deleting unnecessary files**


## Contact

Dr Vinayak Abrol (Professor IIIT-D) _abrol@iiitd.ac.in_


Vaibhav Singh(Research Intern) _vaibhavsinghfcos@gmail.com_


