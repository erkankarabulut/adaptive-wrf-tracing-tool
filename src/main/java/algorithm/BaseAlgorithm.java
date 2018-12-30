package main.java.algorithm;

import main.java.bean.Line;
import main.java.controller.MainController;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

public class BaseAlgorithm {

    public ArrayList<Line> readSparseVector(String sparseVectorPath){
        ArrayList<Line> sparseVector    = new ArrayList<>();
        FileReader fileReader           = null;
        BufferedReader bufferedReader   = null;

        try{
            fileReader          = new FileReader(sparseVectorPath);
            bufferedReader      = new BufferedReader(fileReader);

            String currentLine  = null;
            while ((currentLine = bufferedReader.readLine()) != null){
                String words[]              = currentLine.split(" ");
                ArrayList<Integer> wordList = new ArrayList<>();
                Integer classLabel          = Integer.parseInt(words[0]);

                for(int i=1; i<words.length; i++){
                    wordList.add(Integer.parseInt(words[i].split(":")[0]));
                }

                sparseVector.add(new Line(wordList, classLabel));
            }
        }catch (Exception e){
            e.printStackTrace();
        }

        return sparseVector;
    }

    static void setResults(MainController mainController, Double accuracySum, Double precisionSum, Double recallSum) {
        System.out.println("Done!\n");
        mainController.setAccuracy(accuracySum / mainController.getIterationCountValue());
        mainController.setPrecision(precisionSum / mainController.getIterationCountValue());
        mainController.setRecall(recallSum / mainController.getIterationCountValue());
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

    public Integer getAttributeCount(ArrayList<Line> dataSet){
        Integer attributeCount = new Integer(0);
        for(Line line : dataSet){
            for(Integer attributePointer : line.getWordList()){
                if(attributeCount < attributePointer){
                    attributeCount = attributePointer;
                }
            }
        }

        return attributeCount;
    }

}
