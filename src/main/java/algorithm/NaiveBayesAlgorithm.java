package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.controller.MainController;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class NaiveBayesAlgorithm {

    public SparkBase sparkBase;

    public NaiveBayesAlgorithm(SparkBase sparkBase){
        this.sparkBase = sparkBase;
    }

    public void applyNaiveBayes(String svFilePath, MainController mainController){
        Dataset<Row> dataFrame =
                sparkBase.getSpark().read().format("libsvm").load(svFilePath);
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]
                {mainController.getTrainingDataRate(), mainController.getTestDataRate()}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        NaiveBayes nb = new NaiveBayes();

        NaiveBayesModel model = nb.fit(train);

        Dataset<Row> predictions = model.transform(test);
        predictions.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");

        mainController.setPrecision(evaluator.evaluate(predictions));

        evaluator.setMetricName("weightedRecall");
        mainController.setRecall(evaluator.evaluate(predictions));

        evaluator.setMetricName("accuracy");
        mainController.setAccuracy(evaluator.evaluate(predictions));

    }


}
