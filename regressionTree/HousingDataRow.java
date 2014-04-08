package neu.edu.ml.homework1;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Tadeusz Jordan.
 */
public class HousingDataRow implements Comparable {
    private List<Double> dataRow;

    //determines which feature is being sorted
    private int featureSort;
    private double predictedLabel;


    public double getPredictedLabel() {
        return predictedLabel;
    }

    public void setPredictedLabel(double predictedLabel) {
        this.predictedLabel = predictedLabel;
    }

    public HousingDataRow() {
        this.dataRow = new ArrayList<>();
        this.featureSort = 0;

    }


    public int numberOfFeatures() {
        return dataRow.size() - 1;
    }

    public List<Double> getDataRow() {
        return dataRow;
    }

    public void setFeatureSort(int featureSort) {
        this.featureSort = featureSort;
    }

    public void add(Double element) {
        dataRow.add(element);
        featureSort = dataRow.size() - 1;
    }

    public Double get(int index) {
        return dataRow.get(index);
    }

    public void set(int index, Double element) {
        dataRow.set(index, element);
        featureSort = dataRow.size() - 1;
    }

    public HousingDataRow(String[] tokens) {
        dataRow = new ArrayList<>();


        for(String token : tokens) {
            dataRow.add(Double.parseDouble(token));
        }
        featureSort = dataRow.size() - 1;
    }

    public Double getLabel() {
        return this.dataRow.get(dataRow.size() - 1);
    }

    @Override
    public String toString() {
        return dataRow.toString() + "\n";
    }

    @Override
    public int compareTo(Object o) {
        HousingDataRow housingDataRow = (HousingDataRow) o;

        return (this.get(featureSort) < housingDataRow.get(featureSort) ) ? -1:(this.get(featureSort) > housingDataRow.get(featureSort) ) ? 1:0;
    }

}
