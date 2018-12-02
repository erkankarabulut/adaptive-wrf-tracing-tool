package main.java.algorithm;

import main.java.base.SparkBase;
import main.java.controller.MainController;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class LogisticRegressionAlgorithm {

    public SparkBase sparkBase;
    public String lrFamily;

    public LogisticRegressionAlgorithm(SparkBase sparkBase){
        this.sparkBase  = sparkBase;
        lrFamily        = "binomial";
    }

    public void applyLogisticRegression(String svFilePath, MainController mainController){
        Dataset<Row> dataFrame =
                sparkBase.getSpark().read().format("libsvm").load(svFilePath);
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]
                {mainController.getTrainingDataRate(), mainController.getTestDataRate()}, 1234L);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily(lrFamily);

        LogisticRegressionModel lrModel = lr.fit(training);
        Dataset<Row> predictions = lrModel.transform(test);

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

    public String getLrFamily() {
        return lrFamily;
    }

    public void setLrFamily(String lrFamily) {
        this.lrFamily = lrFamily;
    }
}
