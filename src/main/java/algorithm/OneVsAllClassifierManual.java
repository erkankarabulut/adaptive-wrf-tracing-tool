package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.bean.Line;
import main.java.controller.MainController;
import main.java.util.FileUtil;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.ArrayList;

public class OneVsAllClassifierManual extends BaseAlgorithm {

    public SparkBase sparkBase;
    public FileUtil fileUtil;
    public LogisticRegressionManual logisticRegression;

    public OneVsAllClassifierManual(SparkBase sparkBase){
        this.sparkBase = sparkBase;
        this.fileUtil  = new FileUtil();
        this.logisticRegression = new LogisticRegressionManual(sparkBase);
    }

    public void applyOneVsAllClassifier(MainController controller, String filePath, String fileName, Integer numOfFeatures){

        ArrayList<Double> accuracyList  = new ArrayList<>();
        ArrayList<Double> precisionList = new ArrayList<>();
        ArrayList<Double> recallList    = new ArrayList<>();

        ArrayList<Line> dataSet = readSparseVector(filePath);

        for(int i=0; i<controller.getIterationCountValue(); i++){

            ArrayList<ArrayList<ArrayList<Line>>> datasets = new ArrayList<>();

            Double accuracySumKFold  = new Double(0);
            Double precisionSumKFold = new Double(0);
            Double recallSumKFold    = new Double(0);

            ArrayList<Integer> differentClassLabels     = getDifferentClassLabels(dataSet);
            ArrayList<Line> training                    = null;
            ArrayList<Line> test                        = null;

            int counter;

            if(controller.getTenFold().isSelected()){
                counter = 10;
                datasets = splitAccordingTo10FoldCrossValidation(filePath, i, fileName, numOfFeatures);
            }else {
                counter = 1;
                ArrayList<ArrayList<Line>> splits = splitDateSet(dataSet, controller.getTrainingDataRate());
                training = splits.get(0);
                test     = splits.get(1);
            }

            for(int k=0; k<counter; k++){

                if(controller.getTenFold().isSelected()){
                    training = datasets.get(k).get(0);
                    test     = datasets.get(k).get(1);
                }

                Integer testSetSize = test.size();
                for (Integer classLabel : differentClassLabels){
                    ArrayList<Line> newTraining = reOrganizeDataSet(training, classLabel);
                    ArrayList<Line> newTest     = reOrganizeDataSet(test, classLabel);

                    ArrayList<Double> results = logisticRegression.logisticRegression(newTraining, newTest, numOfFeatures);

                    int tempCount = findCountForClass(test, classLabel);
                    accuracySumKFold  += results.get(0) * (tempCount / testSetSize.doubleValue());
                    precisionSumKFold += results.get(1) * (tempCount / testSetSize.doubleValue());
                    recallSumKFold    += results.get(2) * (tempCount / testSetSize.doubleValue());

                    test = removeDataWithClassLabel(test, classLabel);
                }
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

    public Integer findCountForClass(ArrayList<Line> data, int classLabel){
        int count = 0;

        for(Line line : data){
            if(line.getClassLabel() == classLabel){
                count++;
            }
        }

        return count;
    }

    public ArrayList<Line> removeDataWithClassLabel(ArrayList<Line> dataSet, int classLabel){
        ArrayList<Line> newDateSet = new ArrayList<>();

        for(Line line : dataSet){
            if(line.classLabel != classLabel){
                newDateSet.add(line);
            }
        }

        return newDateSet;
    }

    public ArrayList<Line> reOrganizeDataSet(ArrayList<Line> data, Integer classLabel){
        ArrayList<Line> newDataSet = new ArrayList<>();

        for(Line line : data){
            Line temp = new Line(line.getWordList(), (line.getClassLabel() == classLabel ? 1 : 0));
            newDataSet.add(temp);
        }

        return newDataSet;
    }


}
