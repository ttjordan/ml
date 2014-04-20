package neu.edu.ml.homework1;

/**
 * Created by Tadeusz Jordan.
 */
public class Homework1 {

    public static void main(String [] args) {
        String housingTrainingFile = "/Users/tjordan/Desktop/ML/Data/housing_train.txt";
        String housingTestFile = "/Users/tjordan/Desktop/ML/Data/housing_test.txt";

        Parser parser = new Parser();

        HousingDataRows housingDataTrainingRows = parser.parseFile(housingTrainingFile);
       //housingDataTrainingRows.shiftScaleNormalization();

        HousingDataRows housingDataTestingRows = parser.parseFile(housingTestFile);
        //housingDataTestingRows.shiftScaleNormalization();

        for(int maxTreeDepth = 0; maxTreeDepth < 10; maxTreeDepth++) {
            RegressionTree regressionTree = new RegressionTree(housingDataTrainingRows, maxTreeDepth);
            regressionTree.growTree();
           //regressionTree.printTree();

            regressionTree.predictLabels(housingDataTestingRows);
            double testMse = housingDataTestingRows.calculateMSE();
            regressionTree.predictLabels(housingDataTrainingRows);
            double trainMse = housingDataTrainingRows.calculateMSE();

            System.out.println("maxTreeDepth " + maxTreeDepth + " Test MSE is " + testMse);
            System.out.println("maxTreeDepth " + maxTreeDepth + " Train MSE is " + trainMse);
        }
    }
}
