package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.bean.Line;
import main.java.controller.MainController;
import main.java.util.FileUtil;
import main.java.util.MathUtil;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.ArrayList;

public class LogisticRegressionManual extends BaseAlgorithm {

    private MathUtil mathUtil;
    private SparkBase sparkBase;
    private static Integer ITERATIONS   = 10;
    private static Double LEARNING_RATE = 0.1;
    private FileUtil fileUtil;

    public LogisticRegressionManual(SparkBase sparkBase){
        this.mathUtil  = new MathUtil();
        this.sparkBase = sparkBase;
        this.fileUtil  = new FileUtil();
    }

    public void applyLogisticRegression(MainController controller, String filePath, String fileName, Integer numOfFeatures){

        ArrayList<Double> accuracyList  = new ArrayList<>();
        ArrayList<Double> precisionList = new ArrayList<>();
        ArrayList<Double> recallList    = new ArrayList<>();

        ArrayList<Line> dataSet = readSparseVector(filePath);

        for(int i=0; i<controller.getIterationCountValue(); i++){

            ArrayList<ArrayList<ArrayList<Line>>> datasets = null;
            ArrayList<Line> trainingData = null;
            ArrayList<Line> testData     = null;

            Double accuracySumKFold  = new Double(0);
            Double precisionSumKFold = new Double(0);
            Double recallSumKFold    = new Double(0);

            int counter;

            if(controller.getTenFold().isSelected()){
                counter = 10;
                datasets = splitAccordingTo10FoldCrossValidation(filePath, i, fileName, numOfFeatures);
            }else {
                counter = 1;
                ArrayList<ArrayList<Line>> splits = splitDateSet(dataSet, controller.getTrainingDataRate());
                trainingData = splits.get(0);
                testData = splits.get(1);
            }

            for(int k=0; k<counter; k++){

                if(controller.getTenFold().isSelected()){
                    testData = datasets.get(k).get(0);
                    trainingData = datasets.get(k).get(1);
                }

                ArrayList<Double> results = logisticRegression(trainingData, testData, getAttributeCount(dataSet));

                accuracySumKFold  += results.get(0);
                precisionSumKFold += results.get(1);
                recallSumKFold    += results.get(2);
            }

            accuracySumKFold  /= counter;
            precisionSumKFold /= counter;
            recallSumKFold    /= counter;

            accuracyList.add(accuracySumKFold);
            precisionList.add(precisionSumKFold);
            recallList.add(recallSumKFold);
            System.out.println("Iteration count: " + (i+1));
        }

        setResults(controller, accuracyList, precisionList, recallList);
    }

    public ArrayList<Double> logisticRegression(ArrayList<Line> trainingData, ArrayList<Line> testData, Integer attributeCount){
        ArrayList<Double> generateTrainingModel     = trainModel(trainingData, attributeCount);
        ArrayList<Integer> predictedResults         = new ArrayList<>();
        for(Line testLine : testData){
            if(classify(testLine.wordList, generateTrainingModel) >= 0.5){
                predictedResults.add(1);
            }else {
                predictedResults.add(0);
            }
        }

        return calculateResults(predictedResults, testData);
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

    public double classify(ArrayList<Integer> words, ArrayList<Double> weights) {
        double logit = .0;
        for (int i=0; i<weights.size();i++)  {
            logit += weights.get(i) * (words.contains(i) ? 1 : 0);
        }
        return mathUtil.sigmoid(logit);
    }

}
