# Adaptive Learning-based Tracing Tool for Weather Research and Forecasting Software (2018)

This project includes the implementation of the concepts that I developed in the scope of my BSc. 
thesis at the Software Quality R&D Lab, Yildiz Technical University, Istanbul. Please see the
[thesis](https://github.com/erkankarabulut/adaptive-wrf-tracing-tool/blob/master/doc/Adaptive_Learning_Based_WRF_Tracing_Tool.pdf) itself for more details.

## Abstract 

This study aims to create a tracing tool software for Weather Research and Forecasting
Model[1] based on machine learning algorithms and data provenance, PROV-DM[2]. The
tool provides effectively tracing the internal processes of the model without stopping
the execution of it by reading the log files. Hence, some unwanted situations, for
example faulty or non-accurate input parameters which will then cause the useless
forecasting results, will be detected as soon as they appeared.
Weather Research and Forecasting Model is a well-known and commonly used system in 
weather forecast domain. Main issue about the model is that it can not
be stopped until it finishes its internal processes. Since the model takes long time
to finish its job, knowing the current situation about the model while it works is
crucial.

The WRF model produces very detailed log records which made possible
for us to create a tracing tool in order to trace the internal processes of the model.
Our tracing tool reads the log files of the WRF model and decides whether or
not a certain row includes provenance information. After selecting the rows with
provenance information, the tool decides which provenance relation that the row
has.

For the purpose of selecting the rows with provenance information and deciding the
provenance relation of it, our tracing tool provides Naive Bayes, Logistic Regression,
Decision Tree, Random Forests, Multilayer Perceptron and OneVsAll algorithms.
All of the algorithms are implemented with using Apache Spark’s Machine Learning
library. Also 3 of the algorithms which are Decision Tree, Naive Bayes, Logistic
Regression and OneVsAll classification algorithms are implemented manually.

After deciding the provenance information about the log files, the tool creates a
provenance data file which shows the provenance actors and their relations in PROV-O format[3]. By
using this provenance data file we can easily visualize the content of the provenance
file as nodes and edges thanks to our provenance data visualization tool[4].

### References
[1] Mesoscale and M. M. Laboratory. (2018). Weather research and forecasting model, [Online]. Available: https://www.mmm.ucar.edu/weather-research-and-forecasting-model (visited on 10/20/2018)

[2] W3C. (2013). PROV-DM: The prov data model, [Online]. Available: https://www.w3.org/TR/2013/REC-prov-dm-20130430/ (visited on 10/25/2018).

[3] W3C. (2013). PROV-O: The prov ontology, [Online]. Available: https://www.w3.org/TR/prov-o/ (visited on 10/25/2018)

[4] Yazıcı İ.M., Karabulut E., Aktaş M.S., "A Data Provenance Visualization Approach", The 14th International Conference on Semantics, Knowledge and Grids, Guangzhou, China., 12-14 Sept 2018, pp.1-8
