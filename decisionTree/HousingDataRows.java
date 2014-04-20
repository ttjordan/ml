package neu.edu.ml.homework1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by Tadeusz Jordan.
 */
public class HousingDataRows {
    private List<HousingDataRow> housingDataRowList;
    private double avgLabel;
    //min values in the housingDataRowList, used for shiftScaleNormalization

    public HousingDataRows() {
        this.housingDataRowList = new ArrayList<>();
        this.avgLabel = 0.0;

    }

    public void sort() {
        Collections.sort(housingDataRowList);
    }

    public HousingDataRows(List<HousingDataRow> housingDataRowList) {
        this.housingDataRowList = housingDataRowList;
        updateAvgLabel();
    }

    public void updateAvgLabel() {
        double sum = 0.0;
        for(HousingDataRow housingDataRow : housingDataRowList) {
            sum += housingDataRow.getLabel();
        }

        this.avgLabel = sum / numberOfSamples();
    }

    public HousingDataRow getRow(int x) {
        return housingDataRowList.get(x);
    }

    public List<HousingDataRow> getHousingDataRowList(int from, int to) {
        return housingDataRowList.subList(from, to);
    }

    public int numberOfFeatures() {
        return housingDataRowList.get(0).numberOfFeatures();
    }

    public int numberOfSamples() {
        return housingDataRowList.size();
    }

    public Double getLabel() {
        return avgLabel;
    }

    public void add(HousingDataRow housingDataRow) {
        // update average label
        int numRows = housingDataRowList.size();
        avgLabel = (avgLabel * numRows + housingDataRow.getLabel()) / (numRows + 1);

        housingDataRowList.add(housingDataRow);
        Collections.sort(housingDataRowList);
    }

    public void shiftScaleNormalization() {
        HousingDataRow minHousingDataRow = new HousingDataRow();
        for(int x = 0; x <= numberOfFeatures(); x++) {
            minHousingDataRow.add(Double.MAX_VALUE);
        }

        //calculate min
        for(HousingDataRow housingDataRow : housingDataRowList) {
            for(int x = 0; x < housingDataRow.numberOfFeatures(); x++) {
                if(minHousingDataRow.get(x) > housingDataRow.get(x)) {
                    minHousingDataRow.set(x, housingDataRow.get(x));
                }
            }
        }

        //subtract min
        for(HousingDataRow housingDataRow : housingDataRowList) {
            for(int x = 0; x < housingDataRow.numberOfFeatures(); x++) {
                housingDataRow.set(x, Math.max(0, housingDataRow.get(x) - minHousingDataRow.get(x)));
            }
        }

        HousingDataRow maxHousingDataRow = new HousingDataRow();
        for(int x = 0; x < numberOfFeatures(); x++) {
            maxHousingDataRow.add(Double.NEGATIVE_INFINITY);
        }

        //get max
        for(HousingDataRow row : housingDataRowList) {
            for(int x = 0; x < numberOfFeatures(); x++) {
                if(maxHousingDataRow.get(x) < row.get(x)) {
                    maxHousingDataRow.set(x, row.get(x));
                }
            }
        }

        //divide by max
        for(HousingDataRow row : housingDataRowList) {
            for(int x = 0; x < maxHousingDataRow.numberOfFeatures(); x++) {
                row.set(x, row.get(x) / maxHousingDataRow.get(x));
            }
        }
    }

    public double calculateImpurity() {

        double sumDiff = 0.0;
        for(HousingDataRow housingDataRow : housingDataRowList) {
            sumDiff += Math.pow(housingDataRow.getLabel() - avgLabel , 2);
        }

        return sumDiff/housingDataRowList.size();
    }

    public double calculateMSE() {
        double mse = 0.0;
        for(HousingDataRow housingDataRow : housingDataRowList) {
            mse += Math.pow((housingDataRow.getLabel() - housingDataRow.getPredictedLabel()), 2);
        }

        return mse/housingDataRowList.size();
    }

    public double standardDeviation() {
        return Math.sqrt(calculateMSE());
    }

    public void zeroMeanUnitVarianceNormalization() {
        HousingDataRow avgHousingDataRow = new HousingDataRow();
        for(int x = 0; x < numberOfFeatures(); x++) {
            avgHousingDataRow.add(0.0);
        }

        HousingDataRow varianceHousingDataRow = new HousingDataRow();
        for(int x = 0; x < numberOfFeatures(); x++) {
            varianceHousingDataRow.add(0.0);
        }

        //sum
        for(HousingDataRow row : housingDataRowList) {
            for(int x = 0; x < numberOfFeatures(); x++) {
                avgHousingDataRow.set(x, row.get(x) + avgHousingDataRow.get(x));
            }
        }

        //avg
        for(int x = 0; x < numberOfFeatures(); x++) {
            avgHousingDataRow.set(x, avgHousingDataRow.get(x) / numberOfSamples());
        }

        //variance
        for(int x = 0; x < numberOfFeatures(); x++) {
            double sumDiff = 0.0;
            for(HousingDataRow row : housingDataRowList) {
                sumDiff += Math.pow(row.get(x) - avgHousingDataRow.get(x),2);
            }
            sumDiff = sumDiff / housingDataRowList.size();
            varianceHousingDataRow.set(x, Math.sqrt(sumDiff));
        }

        for(HousingDataRow row : housingDataRowList) {
            for(int x = 0; x < numberOfFeatures(); x++) {
                row.set(x, (row.get(x) - avgHousingDataRow.get(x)) / varianceHousingDataRow.get(x) );
            }
        }
    }
}
