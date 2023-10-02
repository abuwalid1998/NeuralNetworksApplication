package com.ai.neuralnetworks.Controller;


import com.ai.neuralnetworks.Models.InputModel;
import com.ai.neuralnetworks.Models.TrainModel;
import com.ai.neuralnetworks.Services.NeuralNetworkFactory;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/neuralnetwork")
public class Controller {



    NeuralNetworkFactory networkFactory;

    MultiLayerNetwork neuralNetwork;


    public Controller(NeuralNetworkFactory networkFactory) {
        this.networkFactory = networkFactory;
    }


    @GetMapping("/testServer")
    public ResponseEntity<String> TestServer() {

        return new ResponseEntity<>("Welcome to the test server", HttpStatus.OK);

    }


    @PostMapping("/CreateNeuralNetwork")
    public ResponseEntity<String> CreateNeuralNetwork(@RequestBody InputModel model){
        try {

             neuralNetwork = networkFactory.createNeuralNetwork(
                    model.getNumInputs(),
                    model.getNumHidden(),
                    model.getNumOutputs()
            );

            System.out.println("Neural network created successfully :-" + neuralNetwork.summary());

            return new ResponseEntity<>("Neural network is created", HttpStatus.OK);

        }catch (Exception e){
            return new ResponseEntity<>(e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


    @PostMapping("/TrainAndSave")
    public ResponseEntity<String> TrainAndSave(@RequestBody TrainModel model){
        try {

            String saveFilePath = "SavedNetworks/"+networkFactory.generateRandomUsername()+".zip";

            DataSetIterator mnistTrain = new MnistDataSetIterator(model.getBatchSize(), true, 12345);

            networkFactory.trainAndSaveModel(neuralNetwork,mnistTrain,model.getNumEpochs(),saveFilePath);

            System.out.println("Neural network Trained And Saved successfully :-" + neuralNetwork.summary());

            return new ResponseEntity<>("Neural network is Saved and Trained", HttpStatus.OK);

        }catch (Exception e){
            return new ResponseEntity<>(e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/NetworkEvaluation")
    public ResponseEntity<String> TestNeural(@RequestBody String modelpath){
        try {

          String Result = networkFactory.testModel("SavedNetworks/alicejones9.zip");

            return new ResponseEntity<>(Result, HttpStatus.OK);

        }catch (Exception e){
            return new ResponseEntity<>(e.getMessage(), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


}
