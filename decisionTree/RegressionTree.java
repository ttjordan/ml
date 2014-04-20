package neu.edu.ml.homework1;

/**
 * Created by Tadeusz Jordan.
 */
public class RegressionTree {
    private Node root;

    private static final double IMPURITY_THRESHOLD = 0.02;
    private int maxDepth;
    private int trainingDatasetSize;

    private RegressionTree() {
    }

    public RegressionTree(HousingDataRows housingDataRows, int maxDepth) {
        this.root = new Node(housingDataRows);
        this.root.setNodeDepth(0);
        this.maxDepth = maxDepth;
        this.trainingDatasetSize = housingDataRows.numberOfSamples();

    }

    private boolean shouldSplitNode(Node node) {
        if(node.getImpurity() <= IMPURITY_THRESHOLD) {
            return false;
        }
        if(node.getNodeDepth() == maxDepth) {
            return false;
        }
        if(node.numberOfSamples() < trainingDatasetSize * 0.05) {
            return false;
        }

        return true;
    }

    public void predictLabels(HousingDataRows housingDataRows) {
        for(int x = 0; x < housingDataRows.numberOfSamples(); x++) {
            double predictedLabel = root.predictLabel(housingDataRows.getRow(x));

            housingDataRows.getRow(x).setPredictedLabel(predictedLabel);
        }
    }

    private void growTreeHelper(Node node) {
        if(shouldSplitNode(node)) {
            if(node.split()) {
               // System.out.println("parent entropy " + node.getImpurity());
                //System.out.println("left child entropy " + node.getLeftChild().getImpurity());
                //System.out.println("right child entropy " + node.getRightChild().getImpurity());
                growTreeHelper(node.getLeftChild());
                growTreeHelper(node.getRightChild());
            } else {
                node.setIsLeaf(true);
                node.setIsLeaf(true);
            }
        } else {
            node.setIsLeaf(true);
            node.setIsLeaf(true);
        }
    }

    public Node growTree() {
        growTreeHelper(root);
        return root;
    }

    private void printTreeHelper(Node node) {
        for(int x = 0; x < node.getNodeDepth(); x++) {
            System.out.print("----");
        }
        System.out.println(node.toString());
        if(!node.isLeaf()) {
            printTreeHelper(node.getLeftChild());
            printTreeHelper(node.getRightChild());
        }
    }

    public void printTree() {
           printTreeHelper(root);
    }
}
