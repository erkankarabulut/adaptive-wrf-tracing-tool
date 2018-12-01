package main.java.util;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

public class SparseVectorProducerUtil {

    public StructType schema;
    public Dataset<Row> df;
    public JavaSparkContext sc;

    public Integer numOfVocab;

    public SparkConf conf;

    private transient SparkSession spark;

    public SparseVectorProducerUtil(){
        initialize();
    }

    public void initialize(){
        conf = new SparkConf().setAppName("Linear Classifiers Examples")
                .setMaster("local");

        numOfVocab  = new Integer(0);
        sc          = new JavaSparkContext(conf);
        sc.setLogLevel("ERROR");
        spark = SparkSession
                .builder()
                .appName("JavaCountVectorizerExample")
                .getOrCreate();

        schema = new StructType(new StructField[]{
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });

    }

    public String produceSparseVector(String logFilePath) {
        List<Row> data = addToVocabularyList(logFilePath);
        String sparseVectorFilePath = logFilePath + "_sv_" + new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());

        CountVectorizerModel cvModel;

        df = spark.createDataFrame(data, schema);
        cvModel = new CountVectorizer()
                .setBinary(true)
                .setInputCol("text")
                .setOutputCol("feature")
                .setVocabSize(numOfVocab)
                .setMinDF(1)
                .fit(df);

        Dataset<Row> ds = cvModel.transform(df);
        ds.show(false);
        List<Row> listr = ds.collectAsList();

        try {
            FileOutputStream outputStream = new FileOutputStream(sparseVectorFilePath);
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(outputStream);
            BufferedWriter bufferedWriter = new BufferedWriter(outputStreamWriter);

            int[] indices;
            double[] values;

            for (int i = 0; i < listr.size(); i++) {

                SparseVector sv = listr.get(i).getAs(1);
                indices = sv.indices();
                values = sv.values();

                for (int j = 0; j < sv.indices().length; j++) {

                    bufferedWriter.append(" " + (indices[j] + 1) + ":" + (int) values[j]);
                }

                bufferedWriter.newLine();
            }

            bufferedWriter.close();
            spark.stop();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return sparseVectorFilePath;
    }

    public List<Row> addToVocabularyList(String logFilePath) {
        ArrayList<String> vocabList = new ArrayList<String>();
        List<Row> data              = new ArrayList<>();

        try {
            FileReader reader = new FileReader(logFilePath);
            BufferedReader bufferedReader = new BufferedReader(reader);

            String line;
            String[] words;
            while ((line = bufferedReader.readLine()) != null) {
                line = line.trim().replaceAll("\\s{2,}", " ");
                words = line.split(" ");

                for (int i = 0; i < words.length; i++) {
                    if (words[i].contains(":")) {
                        words[i] = words[i].replace(":", "");
                    }
                    vocabList.add(words[i]);
                    numOfVocab++;
                }
                data.add(RowFactory.create((Object) words));
            }

            bufferedReader.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        return data;
    }

}
