package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.controller.MainController;

import main.java.util.FileUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.ArrayList;

public class DecisionTreeAlgorithm extends BaseAlgorithm {

    public SparkBase sparkBase;
    public FileUtil fileUtil;

    public DecisionTreeAlgorithm(SparkBase sparkBase){
        this.sparkBase = sparkBase;
        this.fileUtil  = new FileUtil();
    }

    public void applyDecisionTreeAlgorithm(MainController controller, String filePath, String fileName, Integer numOfFeatures){

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
                datasets = splitAccordingTo10FoldCrossValidation(filePath, i, fileName, sparkBase, numOfFeatures);
            }else {
                counter = 1;
                Dataset<Row>[] splits = fileUtil.getDataSet(sparkBase, filePath).randomSplit(new double[]{controller.getTrainingDataRate(), controller.getTestDataRate()});
                trainingData = splits[0];
                testData = splits[1];
            }

            for(int k=0; k<counter; k++){

                if(controller.getTenFold().isSelected()){
                    testData = datasets.get(k).get(0);
                    trainingData = datasets.get(k).get(1);
                }

                StringIndexerModel labelIndexer = new StringIndexer()
                        .setInputCol("label")
                        .setOutputCol("indexedLabel")
                        .fit(trainingData);

                VectorIndexerModel featureIndexer = new VectorIndexer()
                        .setHandleInvalid("keep")
                        .setInputCol("features")
                        .setOutputCol("indexedFeatures")
                        .setMaxCategories(4)
                        .fit(trainingData);

                DecisionTreeClassifier dt = new DecisionTreeClassifier()
                        .setLabelCol("indexedLabel")
                        .setFeaturesCol("indexedFeatures");

                IndexToString labelConverter = new IndexToString()
                        .setInputCol("prediction")
                        .setOutputCol("predictedLabel")
                        .setLabels(labelIndexer.labels());

                Pipeline pipeline = new Pipeline()
                        .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});

                trainingData.sparkSession().read().format("libsvm");
                PipelineModel model = pipeline.fit(trainingData);

                Dataset<Row> predictions = model.transform(testData);

                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                        .setLabelCol("indexedLabel")
                        .setPredictionCol("prediction")
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
