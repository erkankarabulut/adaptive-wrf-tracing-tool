package main.java.util;

public class MathUtil {

    public double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    public double findInformationGain(Integer x, Integer y){
        return (-(x.doubleValue()/(x.doubleValue()+y.doubleValue()))*log(2,(x.doubleValue()/(x.doubleValue()+y.doubleValue())))) +
                (-(y.doubleValue()/(x.doubleValue()+y.doubleValue()))*log(2,(y.doubleValue()/(x.doubleValue()+y.doubleValue()))));
    }

    public double log(double base, double value){
        return Math.log(value) / Math.log(base);
    }

}
