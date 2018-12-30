package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.controller.MainController;
import main.java.util.MathUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DecisionTreeAlgorithm extends BaseAlgorithm {

    public SparkBase sparkBase;

    public DecisionTreeAlgorithm(SparkBase sparkBase){
        this.sparkBase = sparkBase;
    }

    public void applyDecisionTreeAlgorithm(MainController controller, String filePath){

        Dataset<Row> data = sparkBase
                .getSpark()
                .read()
                .format("libsvm")
                .load(filePath);

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);

        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        Double accuracySum = new Double(0);
        Double precisionSum = new Double(0);
        Double recallSum = new Double(0);
        System.out.println("Here: " + new MathUtil().findInformationGain(9,5));
        for(int i=0; i<controller.getIterationCountValue(); i++){
            Dataset<Row>[] splits = data.randomSplit(new double[]{controller.getTrainingDataRate(), controller.getTestDataRate()});
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];

            DecisionTreeClassifier dt = new DecisionTreeClassifier()
                    .setLabelCol("indexedLabel")
                    .setFeaturesCol("indexedFeatures");

            IndexToString labelConverter = new IndexToString()
                    .setInputCol("prediction")
                    .setOutputCol("predictedLabel")
                    .setLabels(labelIndexer.labels());

            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});

            PipelineModel model = pipeline.fit(trainingData);

            Dataset<Row> predictions = model.transform(testData);

            predictions.select("predictedLabel", "label", "features").show(5);

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("indexedLabel")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");

            accuracySum += evaluator.evaluate(predictions);

            evaluator.setMetricName("weightedPrecision");
            precisionSum += (evaluator.evaluate(predictions));

            evaluator.setMetricName("weightedRecall");
            recallSum += (evaluator.evaluate(predictions));
        }

        setResults(controller, accuracySum, precisionSum, recallSum);
    }

}
