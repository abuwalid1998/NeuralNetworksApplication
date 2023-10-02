package com.ai.neuralnetworks.Services;


import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.Random;

@Service
public class NeuralNetworkFactory {


    public  MultiLayerNetwork createNeuralNetwork(int numInputs, int numHidden, int numOutputs) {
        // Create a configuration for the neural network
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backpropType(BackpropType.Standard)
                .build();

        // Create a MultiLayerNetwork based on the configuration
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);

        model.init();
        return model;
    }

    public  String trainAndSaveModel(MultiLayerNetwork model, DataSetIterator trainingData, int numEpochs, String saveFilePath) throws IOException {
        for (int i = 0; i < numEpochs; i++) {
            while (trainingData.hasNext()) {
                DataSet next = trainingData.next();
                model.fit(next);
            }
            trainingData.reset();
        }

        // Save the trained model to the specified file path
        model.save(new File(saveFilePath));
       return  "Model saved to " + saveFilePath;
    }

    public  String generateRandomUsername() {

        String[] FIRST_NAMES = {"Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Helen"};
        String[] LAST_NAMES = {"Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller", "Wilson"};



        Random random = new Random();

        String randomFirstName = FIRST_NAMES[random.nextInt(FIRST_NAMES.length)];
        String randomLastName = LAST_NAMES[random.nextInt(LAST_NAMES.length)];

        int randomNumber = random.nextInt(1000);
        return randomFirstName.toLowerCase() + randomLastName.toLowerCase() + randomNumber;
    }

    public  String testModel(String modelpath) throws IOException {


        MultiLayerNetwork model = MultiLayerNetwork.load(new File(modelpath), true);

        // Load the MNIST test dataset
        DataSetIterator mnistTest = new MnistDataSetIterator(64, false, 12345);

        Evaluation evaluation = new Evaluation();

        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            evaluation.eval(next.getLabels(), model.output(next.getFeatures()));
        }

       return "Evaluation results:" + evaluation.stats();
    }



}
