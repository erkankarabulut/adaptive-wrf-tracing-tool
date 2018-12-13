package main.java.algorithm;

import main.java.base.Line;
import main.java.controller.MainController;
import main.java.util.MathUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NaiveBayesManual extends BaseAlgorithm{

    MathUtil mathUtil;

    public NaiveBayesManual(){
        mathUtil = new MathUtil();
    }

    // This is a naive bayes classifier implementation for continious attributes
    public void applyNaiveBayesAlgorithmForBinaryClassLabels(MainController controller, String filePath){
        Double accuracy  = new Double(0);
        Double precision = new Double(0);
        Double recall    = new Double(0);

        ArrayList<Line> dataSet = readSparseVector(filePath);
        ArrayList<Double> classLabelRates = findClassLabelRates(dataSet);
        Double zeroRate = classLabelRates.get(0);
        Double oneRate  = classLabelRates.get(1);

        for(int i=0; i<controller.getIterationCountValue(); i++){
            Integer tp = new Integer(0);
            Integer fp = new Integer(0);
            Integer tn = new Integer(0);
            Integer fn = new Integer(0);

            ArrayList<ArrayList<Line>> splittedDataset  = splitDateset(dataSet, controller.getTrainingDataRate(), controller.getTestDataRate());
            ArrayList<Line> training                    = splittedDataset.get(0);
            ArrayList<Line> test                        = splittedDataset.get(1);

            for(Line line : test){
                Double zeroProbability = new Double(1);
                Double oneProbability  = new Double(1);

                for(int k=0; k<line.getWordList().size(); k++){
                    zeroProbability *= findAttributeProbability(training, line.getWordList().get(k), 0);
                }
                zeroProbability *= zeroRate;

                for(int k=0; k<line.getWordList().size(); k++){
                    oneProbability *= findAttributeProbability(training, line.getWordList().get(k), 1);
                }
                oneProbability *= oneRate;

                if(line.getClassLabel() == 0){
                    if(oneProbability > zeroProbability){
                        fp++;
                    }else {
                        tn++;
                    }
                }else {
                    if(oneProbability > zeroProbability){
                        tp++;
                    }else {
                        fn++;
                    }
                }
            }

            accuracy    += ((tp.doubleValue() + tn.doubleValue()) / ((tp.doubleValue() + fp.doubleValue() + fn.doubleValue() + tn.doubleValue())
                    == 0 ? 1 : (tp.doubleValue() + fp.doubleValue() + fn.doubleValue() + tn.doubleValue())));
            precision   += (tp.doubleValue()/((tp.doubleValue() + fp.doubleValue()) == 0 ? 1 : tp.doubleValue() + fp.doubleValue()));
            recall      += (tp.doubleValue()/((tp.doubleValue() + fn.doubleValue()) == 0 ? 1 : (tp.doubleValue() + fn.doubleValue())));
        }

        accuracy  /= controller.getIterationCountValue();
        precision /= controller.getIterationCountValue();
        recall    /= controller.getIterationCountValue();

        controller.setAccuracy(accuracy);
        controller.setPrecision(precision);
        controller.setRecall(recall);
    }

    public Double findAttributeProbability(ArrayList<Line> training, Integer attributeValue, Integer classLabel){
        Integer totalCount   = new Integer(0);
        Integer similarCount = new Integer(0);

        for(Line line : training){
            if(line.getClassLabel() == classLabel){
                totalCount++;
                if(line.getWordList().contains(attributeValue)){
                    similarCount++;
                }
            }
        }

        return (similarCount.doubleValue() / totalCount.doubleValue());
    }

    public ArrayList<ArrayList<Line>> splitDateset(ArrayList<Line> dataset, Integer trainingRate, Integer testRate){
        ArrayList<ArrayList<Line>> splittedDataset  = new ArrayList<>();
        ArrayList<Line> trainingSet                 = new ArrayList<>();
        ArrayList<Line> testSet                     = new ArrayList<>();
        ArrayList<Integer> choosenLines             = new ArrayList<>();
        Random random                               = new Random();
        Integer trainingDatasetSize                 = ((Double) (dataset.size() * (trainingRate.doubleValue() / 100))).intValue();

        for(int i=0; i<trainingDatasetSize; i++){
            Integer linePointer = random.nextInt(dataset.size());
            while (choosenLines.contains(linePointer)){
                linePointer = random.nextInt(dataset.size());
            }

            trainingSet.add(dataset.get(linePointer));
            choosenLines.add(linePointer);
        }

        for(int i=0; i<dataset.size(); i++){
            if(!choosenLines.contains(i)){
                testSet.add(dataset.get(i));
            }
        }

        splittedDataset.add(trainingSet);
        splittedDataset.add(testSet);
        return splittedDataset;
    }

    public ArrayList<Double> findClassLabelRates(ArrayList<Line> dataset){
        ArrayList<Double> classLabelRates = new ArrayList<>();
        Integer zeroCount = new Integer(0);

        for (Line line : dataset){
            if(line.getClassLabel() == 0){
                zeroCount++;
            }
        }

        classLabelRates.add(zeroCount.doubleValue()/dataset.size());
        classLabelRates.add((dataset.size() - zeroCount.doubleValue()) / dataset.size());
        return classLabelRates;
    }
}
