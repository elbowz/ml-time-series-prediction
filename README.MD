# Time series prediction and strategies on stock markets 
### more detailed information in ITALIAN: [REPORT.MD](REPORT.MD)  

## Utilized libs 

* **keras**: neural-network library running on *TensorFlow* 
* **sklearn**: encoding, metrics and classificator models    
* **pandas**:  load and manipulation of **DataSet** 
* **numpy, collections**: advanced data structures  
* **talib**: market *indicators* (*Technical analysis*)    
* **alpha_vantage**: online data market source    
* **json**, **hashlib**, **typing**, **argparse**, **matplotlib**, **os**, etc...  
    
## Usage     
 ```bash    
main.py [-h] [--in-memory-training] [--symbol-data-overwrite] [--force-train] CFG_FILENAME    
```    
    
* **CFG_FILENAME**: path_string - path to config file    
* **--in-memory-training, -m**: TrainingSet manipulation in memory (not suggested). Default *False* 
* **--symbol-data-overwrite, -d**: overwrite the *DataSet* (historical data). Default *False* 
* **--force-train, -f**: force retraining *LSTM*

 > **nota**: major part of configuration is made by config file (see `/cfg` directory)   
 
 ## Credits    
 
 * [LSTM Neural Network for Time Series Prediction](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction) by *Jakob Aungiers*
 * [Alpha Vantage API](https://github.com/RomelTorres/alpha_vantage) by *Romel Torres*
 * [TA-Lib](https://github.com/mrjbq7/ta-lib) by *John Benediktsson*