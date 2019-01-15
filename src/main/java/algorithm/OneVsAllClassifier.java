package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.controller.MainController;
import main.java.util.FileUtil;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.ArrayList;

public class OneVsAllClassifier extends BaseAlgorithm {

    private SparkBase spark;
    private FileUtil fileUtil;

    public OneVsAllClassifier(SparkBase sparkBase){
        this.spark      = sparkBase;
        this.fileUtil   = new FileUtil();
    }

    public void applyOneVsAllClassifier(MainController controller, String filePath, String fileName, Integer numOfFeatures){

        ArrayList<Double> accuracyList  = new ArrayList<>();
        ArrayList<Double> precisionList = new ArrayList<>();
        ArrayList<Double> recallList    = new ArrayList<>();

        for(int i=0; i<controller.getIterationCountValue(); i++){
            Dataset<Row> trainingData = null;
            Dataset<Row> testData     = null;
            ArrayList<ArrayList<Dataset<Row>>> datasets = new ArrayList<>();

            Double accuracySumKFold  = new Double(0);
            Double precisionSumKFold = new Double(0);
            Double recallSumKFold    = new Double(0);

            int counter;

            if(controller.getTenFold().isSelected()){
                counter = 10;
                datasets = splitAccordingTo10FoldCrossValidation(filePath, i, fileName, spark, numOfFeatures);
            }else {
                counter = 1;
                Dataset<Row>[] splits = fileUtil.getDataSet(spark, filePath).randomSplit(new double[]{controller.getTrainingDataRate(), controller.getTestDataRate()});
                trainingData = splits[0];
                testData = splits[1];
            }

            for(int k=0; k<counter; k++) {

                if (controller.getTenFold().isSelected()) {
                    testData = datasets.get(k).get(0);
                    trainingData = datasets.get(k).get(1);
                }

                LogisticRegression classifier = new LogisticRegression()
                        .setMaxIter(10)
                        .setTol(1E-6)
                        .setFitIntercept(true)
                        .setFamily("multinomial");

                OneVsRest ovr = new OneVsRest().setClassifier(classifier);
                OneVsRestModel ovrModel = ovr.fit(trainingData);

                Dataset<Row> predictions = ovrModel.transform(testData)
                        .select("prediction", "label");

                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                        .setMetricName("accuracy");

                accuracySumKFold += evaluator.evaluate(predictions);

                evaluator.setMetricName("weightedPrecision");
                precisionSumKFold += (evaluator.evaluate(predictions));

                evaluator.setMetricName("weightedRecall");
                recallSumKFold += (evaluator.evaluate(predictions));
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
}
