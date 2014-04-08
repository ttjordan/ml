package neu.edu.ml.homework1;

/**
 * Created by Tadeusz Jordan.
 */
public class Node {

    private Node leftChild;
    private Node rightChild;
    private Node parent;
    private HousingDataRows housingDataRows;
    private double impurity;
    private boolean isLeaf;
    private int nodeDepth;

    private Integer featureToSplitOn;
    private double thresholdUseForSplit;

    public Node() {
        this.leftChild = null;
        this.rightChild = null;
        this.parent = null;
        this.housingDataRows = new HousingDataRows();
        this.impurity = 0.0f;
        this.isLeaf = true;
        this.nodeDepth = -1;
    }

    public Node(HousingDataRows housingDataRows) {
        this.leftChild = null;
        this.rightChild = null;
        this.parent = null;
        this.isLeaf = true;
        this.housingDataRows = housingDataRows;
        this.impurity = housingDataRows.calculateImpurity();
        this.nodeDepth = -1;
    }


    public int getNodeDepth() {
        return nodeDepth;
    }

    public void setNodeDepth(int nodeDepth) {
        this.nodeDepth = nodeDepth;
    }

    public boolean isLeaf() {
        return this.isLeaf;
    }

    public void setIsLeaf(boolean isLeaf) {
        this.isLeaf = isLeaf;
    }

    public double getLabel() {
        return housingDataRows.getLabel();
    }

    public int numberOfSamples() {
        return housingDataRows.numberOfSamples();
    }

    public double predictLabel(HousingDataRow housingDataRow) {
        if(this.isLeaf) {
            return this.housingDataRows.getLabel();
        } else {
            if(housingDataRow.get(this.featureToSplitOn) < this.thresholdUseForSplit) {
                 return this.leftChild.predictLabel(housingDataRow);
            } else {
                 return this.rightChild.predictLabel(housingDataRow);
            }
        }
    }

    public boolean split() {
        double minImpurity = this.impurity;
        int minImpurityFeatureIdx = -1;
        int minImpuritySplitIdx = -1;

        // find optimal split and set parents and children
        for(Integer featureIdx = 0; featureIdx < this.housingDataRows.numberOfFeatures(); featureIdx++) {
            for(int x = 0; x < this.housingDataRows.numberOfSamples(); x++) {
                this.housingDataRows.getRow(x).setFeatureSort(featureIdx);
            }
            this.housingDataRows.sort();

            for(int splitIdx = 1; splitIdx < this.housingDataRows.numberOfSamples(); splitIdx++) {
                HousingDataRows left = new HousingDataRows(
                        this.housingDataRows.getHousingDataRowList(0, splitIdx));
                HousingDataRows right = new HousingDataRows(
                        this.housingDataRows.getHousingDataRowList(splitIdx, this.housingDataRows.numberOfSamples()));
                int leftSize = left.numberOfSamples();
                int rightSize = right.numberOfSamples();
                double leftImpurity =   left.calculateImpurity();
                double rightImpurity = right.calculateImpurity();
                double candidateImpurity = (leftImpurity*leftSize + rightSize*rightImpurity) / (leftSize+rightSize);
                if(candidateImpurity < minImpurity) {
                    minImpurity = candidateImpurity;
                    minImpurityFeatureIdx = featureIdx;
                    minImpuritySplitIdx = splitIdx;
                }
            }
        }
        if(minImpurity != this.impurity) {
            if(minImpurityFeatureIdx != -1) {

            }
            if(minImpuritySplitIdx != -1) {
                for(int x = 0; x < this.housingDataRows.numberOfSamples(); x++) {
                    this.housingDataRows.getRow(x).setFeatureSort(minImpurityFeatureIdx);
                }
                this.housingDataRows.sort();

                HousingDataRows left = new HousingDataRows(
                        this.housingDataRows.getHousingDataRowList(0, minImpuritySplitIdx));
                HousingDataRows right = new HousingDataRows(
                        this.housingDataRows.getHousingDataRowList(minImpuritySplitIdx, this.housingDataRows.numberOfSamples()));
                this.leftChild = new Node(left);
                this.leftChild.setParent(this);
                this.leftChild.setNodeDepth(this.nodeDepth + 1);

                this.rightChild = new Node(right);
                this.rightChild.setParent(this);
                this.rightChild.setNodeDepth(this.nodeDepth + 1);

                this.featureToSplitOn = minImpurityFeatureIdx;
                this.thresholdUseForSplit = housingDataRows.getRow(minImpuritySplitIdx).get(minImpurityFeatureIdx);

                this.isLeaf = false;
                return true;
            }
        }
        return false;
    }


    public Node getLeftChild() {
        return leftChild;
    }

    public void setLeftChild(final Node leftChild) {
        this.leftChild = leftChild;
    }

    public Node getRightChild() {
        return rightChild;
    }

    public void setRightChild(final Node rightChild) {
        this.rightChild = rightChild;
    }

    public Node getParent() {
        return parent;
    }

    public void setParent(final Node parent) {
        this.parent = parent;
    }

    public HousingDataRows getHousingDataRows() {
        return housingDataRows;
    }

    public void setHousingDataRows(final HousingDataRows housingDataRows) {
        this.housingDataRows = housingDataRows;
    }

    public double getImpurity() {
        return impurity;
    }

    public void setImpurity(final double impurity) {
        this.impurity = impurity;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append(" isleaf is : " + this.isLeaf);
        result.append(" featureToSplitOn is : " + this.featureToSplitOn);
        result.append(" thresholdUsefForSplit is : " + this.thresholdUseForSplit);
        result.append(" Impurity is : " + this.impurity);
        result.append(" NodeDepth is : " + this.nodeDepth);
        result.append(" nodeSize is : " + this.getHousingDataRows().numberOfSamples());
        if(this.isLeaf) {
            result.append(" Label is : " + this.getLabel());
        }
        result.append("\n");

        return result.toString();
    }
}
