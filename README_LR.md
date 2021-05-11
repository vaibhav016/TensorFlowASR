# Welcome to experiments on low rank approximation on speech 
This project is started and guided by Dr Vinayak Abrol.(1 April 2021) 

###Steps to be followed for setup 

1) Below are links of Research papers that we must read before proceeding further 
    1) http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf
        _this is sir's paper and new idea of low rank filtering which we need to implement._
    
    2) https://arxiv.org/pdf/2005.03191.pdf  _this is contextNet, over which we need to implement new ideas._
    
2) Links to Dataset  https://drive.google.com/drive/folders/1jf0wWOZ59YmOziRL37cQLlaPUys_DlqC?usp=sharing 
_This link has all the tar.gz dataset files, that would be needed. (7.5 gb)_
    1) dev-clean
    2) dev-other
    3) train-clean-100
    4) test-clean
    
3) We have access to quadro rtx 4000 VM. Download Anydesk software. For login credentials request sir.
    
4) There is a conda env on the machine. 
    1) **tf2** 
    
5) Check for the environment.yml file in the project directory, if yes, then cool, if not , then Download the environment.yml and store in the current working directory, https://drive.google.com/file/d/1EbZgWls771POqJNhuWt9P62xFKQSiUwz/view?usp=sharing
There is another file config.yml and create_transcripts.sh which will be used in the code further.  

6) For testing purposes create a new environment
    1) conda create -n {env_name} -f {path to environment.yml}
    2) conda activte {env_name}
    3) Setup of TensorflowASR 
        1) `https://github.com/vaibhav016/TensorFlowASR.git`
        2) do this git clone in the current directory. Lets say we are in /Desktop 
        3) Go through the README of the project. 
        4) Our current objective is to implement low rank conv module in the contextNet model
        5) Do a `git checkout gcp_lr_cn`
        6) This branch by default implements LRCN but the config path are according to the gcp machine. Those will have to be changed.
    
    4) Now go to /TensorFlowASR/scripts/Datasets and run `sh donwload_links.sh`. Take care of spelling. 
    5) If only some testing needs to be done, kindly comment the 360 and 500 datasets. They take a lot of time to download. The essential are 
        1) test-clean, 
        2) train-clean-100
        3) dev-clean

 
7) Now do your changs with the model or any other files and once satisfied, build the model by follwoing command, go to /Desktop/TensorFlowASR and run this command
    1) `python setup.py build`
    2) `python setup.py install`
    2) `sh create_transcripts_from_data.sh`
    this will create transcripts. 

8) Now repalce the config.yml in Desktop/TensorFlowASR/examples/contextnet/ with the one donwloaded from link
We need to replace the datasets path here, according to your machine. Thats all

9) Just to be double-sure, this project gives a setup file too. Lets run it to be absolutely sure. 
    1) `python3  Desktop/TensorFlowASR/setup.py build`
    2) `python3  Desktop/TensorFlowASR/setup.py install`
    3) **WORD OF CAUTION**- whenever any change is made to the code and scripts are run, then you would need to run the above two commands first, because setuptools install
    the packages in env/lib/site-packages/python3.8/XXXX. So your change will not be reflected unless you build and install again. 

10) Since our requirements are already there, so it would skip and process would be finished with success.

11) Finally we will run our main script.
    1)`python3 /Desktop/TensorFlowASR/examples/contextnet/train.py`
    
    2) since we need logging into files and in background to free terminal so better run this command
     `_nohup python3 /Desktop/TensorFlowASR/examples/contextnet/train.py > logging.out &_`
     
    3) if you want error in a seperate file then 
         `_nohup python3 /Desktop/TensorFlowASR/examples/contextnet/train.py > logging.out 2>&1 &_`

12) For testing run this script 
    1) `python3 /Desktop/TensorFlowASR/examples/contextnet/test.py --saved /Desktop/TensorFlowASR/examples/contextnet/latest.h5`

13) _**when we make any changes in model.py or any other file, we would need to build this project again by running this command** _
    1) `python setup.py build` 
    2) `python setup.py install`

14) Suppose we want to retrain our model after making some changes. Then we would first need to run step 13) and then would need to remove the checkpoint and tensorboard folders by running these commands
    1) `rm -r Desktop/TensorFLowASR/examples/contextnet/checkpoints`
    2) `rm -r Desktop/TensorFLowASR/examples/contextnet/tensorboard`
    3)  if testing is need to be done again, then delete the test.tsv file too and provide the path to new model.h5
     



## Contact

Dr Vinayak Abrol (Professor IIIT-D) _abrol@iiitd.ac.in_


Vaibhav Singh(Research Intern) _vaibhavsinghfcos@gmail.com_
