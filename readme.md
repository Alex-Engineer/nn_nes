### git init 

### conda packages
conda install numpy scipy pandas matplotlib scikit-learn  seaborn jupyter -y 
conda install  pytorch torchvision torchaudio -c pytorch -y
conda install tensorboard


If you want to make your environment file work across platforms, you can use the conda env export --from-history 

# Что еще можно сделать

1) Data
    * 4 types of random data
    * Dataset and dataloader
    * add data preprocessing (StandardScaler())
    * train/test split
    * add additional features
    
2) NN architecture
    * classification  (**regression)
    * n of hidden layers and neurons; activations 
    * output 
    
3) NN training/testing
    * loss function
    * optimizer 
    * n of epochs 
    * calculate metrics
    * save every epoch
    * load from saved
    
4) Visualization 
    * plot data
    * show train/test data 
    * discretize output
    * loss plots 
    * tensorboard
    
5) Add gpu support



Run tensorboard 
tensorboard --logdir runs