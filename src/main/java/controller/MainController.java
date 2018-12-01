package main.java.controller;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import main.java.util.ClassLabelProducerUtil;
import main.java.util.SparseVectorProducerUtil;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.net.URL;
import java.util.ResourceBundle;

public class MainController implements Initializable {

    @FXML Button chooseLogFileButton;
    @FXML Button runButton;

    @FXML ComboBox selectAlgorithmComboBox;

    private String logFileName;
    private String logFilePath;
    private static final String filteredLogFilePath = System.getProperty("user.dir") + "/data/filtered_log_file";

    private Integer algorithmPointer;

    public ClassLabelProducerUtil classLabelProducerUtil;
    public SparseVectorProducerUtil sparseVectorProducerUtil;

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        classLabelProducerUtil   = new ClassLabelProducerUtil();
        sparseVectorProducerUtil = new SparseVectorProducerUtil();

        chooseLogFileButton.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {
                JFileChooser chooser = new JFileChooser();
                if(chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                    logFileName = chooser.getSelectedFile().getName();
                    logFilePath = chooser.getSelectedFile().getPath();
                }
            }
        });

        selectAlgorithmComboBox.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {
                algorithmPointer = selectAlgorithmComboBox.getSelectionModel().getSelectedIndex();
            }
        });

        runButton.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent actionEvent) {

                if(algorithmPointer == 0){       // Logistic Regression
                    String sparseVectorFilePath = sparseVectorProducerUtil.produceSparseVector(logFilePath);
                    classLabelProducerUtil.produceBinaryLabels(logFilePath, sparseVectorFilePath, filteredLogFilePath);
                    classLabelProducerUtil.produceMulticlassLabels(logFilePath, sparseVectorFilePath, filteredLogFilePath);

                }else if(algorithmPointer == 1){ // Bayesian Linear Regression

                }else if(algorithmPointer == 2){ // Least Absolute Deviation Regression

                }else if(algorithmPointer == 3){ // Winnow Algorithm

                }
            }
        });
    }

}
