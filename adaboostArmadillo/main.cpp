//
//  main.cpp
//  adaboostecoc
//
//  Created by Tad Jordan on 4/26/14.
//  Copyright (c) 2014 Tad Jordan. All rights reserved.
//

#include <iostream>
#include "/usr/include/armadillo"
#include <limits>
#include <stdlib.h>
#include <time.h>
#include <set>
#include <map>

using namespace std;
using namespace arma;

struct Stump {
    int dimension;
    float threshold;
    float error;
    int less;
    int greater;
};
static bool cachedThresholds = false;

struct Info {
    int column;
    int row;
    int threshold;
    
    bool operator==(const Info& r) const {
        return ((column == r.column) && (row == r.row) && (threshold == r.threshold));
    }
    bool operator<(const Info& r) const {
        if((column < r.column) || ((column == r.column) && (row < r.row))) {
            return true;
        } else {
            return false;
        }
    }
};



Stump build_stump(mat data, vec labels, vec D) {
    int numCols = data.n_cols;
    int numRows = data.n_rows;
    
    int minStumpError = numeric_limits<int>::max();
    Stump minErrorStump;
    
    for(int col = 0; col < numCols; ++col) {
        cout << "building stump for column " << col << endl;
        double errorGreater = numeric_limits<int>::max();
        int minErrorGreaterThreshold = numeric_limits<int>::max();
        vec yPredictGreater = zeros<vec>(numRows);
        yPredictGreater.fill(-1);
        
        double errorLess = numeric_limits<int>::max();
        int minErrorLessThreshold = numeric_limits<int>::max();
        vec yPredictLess = zeros<vec>(numRows);
        yPredictLess.fill(-1);
        
        static std::map<Info, vec> mapGreater;
        static std::map<Info, vec> mapLess;
        for(int rowThreshold = 0; rowThreshold < 2; ++rowThreshold) {
            for(int innerRow = 0; innerRow < numRows; ++innerRow) {
                Info cacheInfo;
                cacheInfo.column = col;
                cacheInfo.row = innerRow;
                cacheInfo.threshold = rowThreshold;
                if(!cachedThresholds) {
                    uvec thresholdIdxGreater = find(data.col(col) >= rowThreshold);
                    for(int thresholdIdxxGreater : thresholdIdxGreater) {
                        yPredictGreater(thresholdIdxxGreater) = 1;
                    }
                    
                    uvec thresholdIdxLess = find(data.col(col) < rowThreshold);
                    for(int thresholdIdxxLess: thresholdIdxLess) {
                        yPredictLess(thresholdIdxxLess) = 1;
                    }
                    
                    mapGreater[cacheInfo] = yPredictGreater;
                    mapLess[cacheInfo] = yPredictLess;
                } else {
                    yPredictGreater = mapGreater[cacheInfo];
                    yPredictLess = mapLess[cacheInfo];
                }
            }
            double tempErrorGreater = 0.0;
            double tempErrorLess = 0.0;
            for(int r = 0; r < numRows; ++r) {
                if(labels(r) != yPredictGreater(r)) {
                    tempErrorGreater += D(r);
                }
                if(labels(r) != yPredictLess(r)) {
                    tempErrorLess += D(r);
                }
            }
            tempErrorGreater = tempErrorGreater / sum(D);
            tempErrorLess = tempErrorLess / sum(D);
            
            if(tempErrorGreater < errorGreater) {
                errorGreater = tempErrorGreater;
                minErrorGreaterThreshold = rowThreshold;
            }
            if(tempErrorLess < errorLess) {
                errorLess = tempErrorLess;
                minErrorLessThreshold = rowThreshold;
            }
        }
        
        if(errorLess >= errorGreater) {
            if(minStumpError > errorGreater) {
                minErrorStump.dimension = col;
                minErrorStump.threshold = minErrorGreaterThreshold;
                minErrorStump.error = errorGreater;
                minErrorStump.greater = 1;
                minErrorStump.less = -1;
                
                minStumpError = errorGreater;
            }
        } else {
            if(minStumpError > errorLess) {
                minErrorStump.dimension = col;
                minErrorStump.threshold = minErrorLessThreshold;
                minErrorStump.error = errorLess;
                minErrorStump.greater = 1;
                minErrorStump.less = -1;
                
                minStumpError = errorLess;
            }
        }
    }
    cachedThresholds = true;
    
    return minErrorStump;
}

vec predStump(mat data, Stump stump) {
    vec label = zeros<vec>(data.n_rows);
    
    for(int row = 0; row < data.n_rows; ++row) {
        if(data.col(stump.dimension).at(row, 0) >= stump.threshold) {
            label(row) = stump.greater;
        } else {
            label(row) = stump.less;
        }
    }
    
    return label;
}

int main(int argc, const char * argv[])
{
    cout << "Armadillo version: " << arma_version::as_string() << endl;
    
    mat trainFull;
    cout << "loading training set" << endl;
    trainFull.load("/Users/tjordan/Desktop/20newsgroup/parsed_train.txt",csv_ascii);
    cout << "train rows " << trainFull.n_rows << " cols " << trainFull.n_cols << endl;
    cout << "done" << endl;
    mat ecoc;
    cout << "loading ecoc codes" << endl;
    ecoc.load("/Users/tjordan/Desktop/20newsgroup/ecoccodes.csv",csv_ascii);
    cout << "done" << endl;
    mat test;
    cout << "loading test set" << endl;
    test.load("/Users/tjordan/Desktop/20newsgroup/parsed_test.txt",csv_ascii);
    
    cout << "successfully loaded data" << endl;
    
    const int PERCENT = 10;
    const int NUMITERATIONS = 20;
    
    int numRows = round(PERCENT * trainFull.n_cols / 100);
    srand(time(0)); //initialize the random seed
    int tempNum;
    
    set<int> indicesSubset;
    uvec rowIndicesForSubset = uvec(numRows);
    int counter = 0;
    while(true) {
        tempNum = rand() % trainFull.n_rows;
        if(indicesSubset.find(tempNum) == indicesSubset.end()) {
            indicesSubset.insert(tempNum);
            rowIndicesForSubset.at(counter) = tempNum;
            counter++;
            if(counter == numRows) {
                break;
            }
        }
    }
    uvec colIndices = uvec(trainFull.n_cols);
    for(int tt = 0; tt < trainFull.n_cols; ++tt) {
        colIndices(tt) = tt;
    }
    cout << max(rowIndicesForSubset) << " " << max(colIndices) << endl;
    mat train = trainFull.submat(rowIndicesForSubset, colIndices);
    
    
    int numberOfClassifiers = ecoc.n_cols;
    int numberOfClasses = 8;
    
    vec originalLabels = train.col(0);
    mat trainingSetData = train.submat(0, 1, train.n_rows - 1, train.n_cols - 1);
    
    
    vector<vector<Stump> > abClassifiers;
    vector<vec> abClassifiersWeights;
    
    for(int classifier = 0; classifier < numberOfClassifiers; ++classifier) {
        cout << "Training classifier " << classifier << endl;
        vec newTrainingLabels(trainingSetData.n_rows);
        newTrainingLabels.fill(-1);
        
        for(int j = 0; j < numberOfClasses; ++j) {
            if(ecoc(j, classifier) == 1) {
                for(int row = 0; row < trainingSetData.n_rows; ++row) {
                    uvec indices = find(originalLabels == j);
                    for(int i : indices) {
                        newTrainingLabels(i) = 1;
                    }
                }
            }
        }
        vec errorsAb = vec(NUMITERATIONS);
        vec D = vec(trainingSetData.n_rows);
        D.fill((float)1/trainingSetData.n_rows);
        vector<Stump> weakClassifiers;
        vec weakClassifiersAlpha = vec(NUMITERATIONS);
        for(int iter = 0; iter < NUMITERATIONS; ++iter) {
            cout << "Training classifier " << classifier << " iteration " << iter << endl;
            weakClassifiers.push_back(build_stump(trainingSetData, newTrainingLabels, D));
            weakClassifiersAlpha(iter) = 0.5 * log((1 - weakClassifiers.at(iter).error) / weakClassifiers.at(iter).error);
            // update D
            vec weakClassifierPred = predStump(trainingSetData, weakClassifiers.at(iter));
            vec tempD = -1 * weakClassifiersAlpha(iter) * (newTrainingLabels % weakClassifierPred);
            tempD = D % exp(tempD);
            D = tempD / sum(tempD);
        }
        abClassifiers.push_back(weakClassifiers);
        abClassifiersWeights.push_back(weakClassifiersAlpha);
    }
    
    vec validationSetLabels = test.col(0);
    mat validationSetData = test.submat(0, 1, test.n_rows - 1, test.n_cols - 1);
    
    cout << "computing testing error" << endl;
    
    mat labels;
    labels.zeros(validationSetData.n_rows, numberOfClassifiers);
    for(int i = 0; i < numberOfClassifiers; ++i) {
        cout << "classifier " << i << endl;
        vector<vec> WCLabels;
        vec weakClassifiersAlphaa = abClassifiersWeights.at(i);
        vector<Stump> weakClassifierr = abClassifiers.at(i);
        for(int wc = 0; wc < NUMITERATIONS; ++wc) {
            WCLabels.push_back(weakClassifiersAlphaa(wc) * predStump(validationSetData, weakClassifierr.at(wc)));
        }
        vec WCLabelsSums = vec(WCLabels.size());
        for(int wclabel = 0; wclabel < WCLabels.size(); ++wclabel) {
            WCLabelsSums.at(wclabel) = sum(WCLabels.at(wclabel));
        }
        uvec wcindices = find(WCLabelsSums > 0);
        vec ablabel = vec(validationSetData.n_rows);
        ablabel.fill(-1);
        for(int idx : wcindices) {
            ablabel(idx) = 1;
        }
        labels.col(i) = ablabel;
    }
    vec finalLabels = vec(validationSetData.n_rows);
    cout << "computing final labels" << endl;
    for(int row = 0; row < validationSetData.n_rows; ++row) {
        int closestClass = numeric_limits<int>::max();
        int closestDistance = numeric_limits<int>::max();
        for(int classIdx = 0; classIdx < numberOfClasses; ++classIdx) {
            double tempDistance = sum(labels.row(row) != ecoc.row(classIdx));
            if(tempDistance < closestDistance) {
                closestDistance = tempDistance;
                closestClass = classIdx;
            }
        }
        finalLabels.at(row) = closestClass;
    }
    uvec abErrorIdx = finalLabels != validationSetLabels;
    double tempSum = sum(abErrorIdx);
    double abErrorValidation = tempSum / validationSetData.n_rows;
    
    cout << "Testing accuracy " << 1.0 -  abErrorValidation << endl;
    
    
    
    return 0;
}

