package main.java.algorithm;

import main.java.bean.Line;
import main.java.controller.MainController;
import main.java.util.MathUtil;

import java.util.ArrayList;

public class LogisticRegressionManual extends BaseAlgorithm {

    private MathUtil mathUtil;
    private static Integer ITERATIONS   = 10;
    private static Double LEARNING_RATE = 0.1;

    public LogisticRegressionManual(){
        this.mathUtil = new MathUtil();
    }

    public void applyLogisticRegression(MainController controller, String filePath){
        Double accuracy  = new Double(0);
        Double precision = new Double(0);
        Double recall    = new Double(0);

        ArrayList<Line> dataSet = readSparseVector(filePath);

        for(int i=0; i<controller.getIterationCountValue(); i++){
            ArrayList<ArrayList<Line>> splittedDataset  = splitDateset(dataSet, controller.getTrainingDataRate(), controller.getTestDataRate());
            ArrayList<Line> training                    = splittedDataset.get(0);
            ArrayList<Line> test                        = splittedDataset.get(1);

            ArrayList<Double> generateTrainingModel     = trainModel(training, getAttributeCount(dataSet));
            ArrayList<Integer> predictedResults         = new ArrayList<>();
            for(Line testLine : test){
                if(classify(testLine.wordList, generateTrainingModel) >= 0.5){
                    predictedResults.add(1);
                }else {
                    predictedResults.add(0);
                }
            }

            ArrayList<Double> results = calculateResults(predictedResults, test);
            accuracy    += results.get(0);
            precision   += results.get(1);
            recall      += results.get(2);
        }

        accuracy    /= controller.getIterationCountValue();
        precision   /= controller.getIterationCountValue();
        recall      /= controller.getIterationCountValue();

        controller.setAccuracy(accuracy);
        controller.setPrecision(precision);
        controller.setRecall(recall);

    }

    public ArrayList<Double> calculateResults(ArrayList<Integer> predictedResults, ArrayList<Line> testData){
        ArrayList<Double> results = new ArrayList<>();
        Integer tp = 0;
        Integer fp = 0;
        Integer tn = 0;
        Integer fn = 0;

        for(int i=0; i<predictedResults.size(); i++){
            if(testData.get(i).getClassLabel() == 0){
                if(predictedResults.get(i) == 0){
                    tn++;
                }else {
                    fn++;
                }
            }else {
                if(predictedResults.get(i) == 0){
                    fn++;
                }else {
                    tp++;
                }
            }
        }

        Double accuracy     = (tp.doubleValue() + tn.doubleValue()) / testData.size();
        Double precision    = (tp.doubleValue()) / (tp.doubleValue() + fp.doubleValue());
        Double recall       = (tp.doubleValue()) / (tp.doubleValue() + fn.doubleValue());

        results.add(accuracy);
        results.add(precision);
        results.add(recall);

        return results;
    }

    public ArrayList<Double> trainModel(ArrayList<Line> trainSet, Integer attributeCount){
        ArrayList<Double> weights = new ArrayList<>();
        for (int n=0; n<ITERATIONS; n++) {
            for (int i=0; i<trainSet.size(); i++) {
                ArrayList<Integer> words    = trainSet.get(i).getWordList();
                Double predictedValue       = classify(words, weights);
                Integer label               = trainSet.get(i).getClassLabel();

                for (int j=1; j<=attributeCount; j++) {
                    if(weights.size() < j+1){
                        weights.add(new Double(0));
                    }

                    weights.set(j-1, weights.get(j-1) + LEARNING_RATE * (label - predictedValue) * (words.contains(j-1) ? 1 : 0));
                }
            }

        }

        return weights;
    }

    private double classify(ArrayList<Integer> words, ArrayList<Double> weights) {
        double logit = .0;
        for (int i=0; i<weights.size();i++)  {
            logit += weights.get(i) * (words.contains(i) ? 1 : 0);
        }
        return mathUtil.sigmoid(logit);
    }

}
