package main.java.algorithm;

import main.java.bean.Line;
import main.java.bean.Node;
import main.java.controller.MainController;
import main.java.util.MathUtil;

import java.util.ArrayList;

public class DecisionTreeAlgorithmManual extends BaseAlgorithm {

    public void applyDecisionTree(MainController controller, String filePath){
        Double accuracy  = new Double(0);
        Double precision = new Double(0);
        Double recall    = new Double(0);

        ArrayList<Line> dataSet = readSparseVector(filePath);
        for(int i=0; i<controller.getIterationCountValue(); i++) {
            Integer tp = new Integer(0);
            Integer fp = new Integer(0);
            Integer tn = new Integer(0);
            Integer fn = new Integer(0);

            ArrayList<ArrayList<Line>> splittedDataset  = splitDateset(dataSet, controller.getTrainingDataRate(), controller.getTestDataRate());
            ArrayList<Line> training                    = splittedDataset.get(0);
            ArrayList<Line> test                        = splittedDataset.get(1);

            Double baseInfoGain                         = findBaseInfoGain(training);
            Node rootNode                               = findRootNode(training, getAttributeCount(training));

            Boolean shouldStop = false;
            while (!shouldStop){


            }



        }

    }

    public Double findBaseInfoGain(ArrayList<Line> data){
        Integer positiveClassCount = 0;
        for(Line line : data){
            if(line.getClassLabel() == 1){
                positiveClassCount++;
            }
        }

        return new MathUtil().findInformationGain(positiveClassCount, (data.size() - positiveClassCount));
    }

    public Node findRootNode(ArrayList<Line> data, Integer attributeCount){
        Node node           = new Node();
        MathUtil mathUtil   = new MathUtil();

        Double infoGain                 = new Double(0);
        int attributeWithMaxInfoGain    = 0;
        for(int i=1; i<=attributeCount; i++){
            Integer plc = 0; // Positive label count
            Integer plcForplc = 0; // Positive label count for positive class labels
            Integer plcFornlc = 0; // Positive label count for negative class labels
            for(Line line : data){
                if(line.getWordList().contains(i)){
                    plc++;
                    if(line.getClassLabel() == 1){
                        plcForplc++;
                    }
                }else {
                    if(line.getClassLabel() == 1){
                        plcFornlc++;
                    }
                }
            }

            Double temp = ((plc.doubleValue() / (data.size())) * mathUtil.findInformationGain(plcForplc, (plc - plcForplc))) +
                    (((data.size() - plc.doubleValue()) / (data.size())) * mathUtil.findInformationGain(plcFornlc, ((data.size() - plc) - plcFornlc)));
            if(infoGain <= temp){
                infoGain = temp;
                attributeWithMaxInfoGain = i;
            }
        }

        node.setAttributePointer(attributeWithMaxInfoGain);
        return node;
    }

}
