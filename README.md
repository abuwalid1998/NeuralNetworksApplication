# NeuralNetworksApplication

Neural Network Programming , Simple Application Shows how to Create ,Save and Evaluate the Neural Network using  MINSTDATA

## 🔧 Technologies & Tools

- Programming Languages: Java - 11
- Backend Frameworks: Spring Boot
- Version Control: Git
- Deeplearning4j



## API Reference

#### Create Neural Network

```http
  POST neuralnetwork/CreateNeuralNetwork
```

#### Input
```json

{
    "numInputs":784,
    "numHidden":250,
    "numOutputs":10
}

```

#### Output
```json

 Neural network is created

```
#### Train And Save Network

```http
  POST /neuralnetwork/TrainAndSave
```
#### Input
```json

{
    "batchSize":64,
    "numEpochs":10
}

```

#### Output
```json

 Neural network is Saved and Trained

```

#### Network Evaluation

```http
  POST /neuralnetwork/NetworkEvaluation
```


#### Input 
```json

{
 "modelpath": "C:/Users/PC/OneDrive/Desktop/NeuralNetworks/SavedNetworks/alicejones9.zip"
}


```
#### Output 
```json

Evaluation results:

========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0.8848
 Precision:       0.8835
 Recall:          0.8830
 F1 Score:        0.8828
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


=========================Confusion Matrix=========================
    0    1    2    3    4    5    6    7    8    9
---------------------------------------------------
  945    0    7    2    0    6   16    1    3    0 | 0 = 0
    0 1098    7    4    1    1    4    0   20    0 | 1 = 1
   19   14  865   21   17    1   26   23   43    3 | 2 = 2
    6    2   23  888    1   33    6   17   22   12 | 3 = 3
    3    8    7    0  885    3   11    2    9   54 | 4 = 4
   23   12    9   63   19  694   22   10   27   13 | 5 = 5
   19    3    8    2   15   19  888    2    2    0 | 6 = 6
    3   23   32    3   11    2    1  907    6   40 | 7 = 7
    8   10   12   43   13   27   13   11  815   22 | 8 = 8
   13   10   12   12   48   16    0   25   10  863 | 9 = 9

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================


```
