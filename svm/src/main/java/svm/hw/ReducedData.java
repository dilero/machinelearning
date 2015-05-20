package svm.hw;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;

public class ReducedData {
	public static void main(String[] args) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(
				"src/main/resources/optdigits01pca2att.arff"));
		Instances trainingData = new Instances(reader);
		reader.close();
		// setting class attribute
		trainingData.setClassIndex(trainingData.numAttributes() - 1);


		 String[] linear = weka.core.Utils.splitOptions("-K 0");
		 String[] polynomial = weka.core.Utils.splitOptions("-K 1");
		 String[] radial = weka.core.Utils.splitOptions("-K 2");
		 String[] sigmoid = weka.core.Utils.splitOptions("-K 3");

		 LibSVM svm = new LibSVM();
		 svm.setOptions(linear);
		 
		 Evaluation eval2 = new Evaluation(trainingData);
		 System.out.println("******linear*********************");
		 eval2.crossValidateModel(svm, trainingData, 10, new Random(1));
		 System.out.println("*****Estimated Accuracy: "
		 + Double.toString(eval2.pctCorrect()));
		 System.out.println("*************polynomial**********");
		 svm.setOptions(polynomial);
		 eval2.crossValidateModel(svm, trainingData, 10, new Random(1));
		 System.out.println("*****Estimated Accuracy: "
		 + Double.toString(eval2.pctCorrect()));
		 System.out.println("*************radial**********");
		 svm.setOptions(radial);
		 eval2.crossValidateModel(svm, trainingData, 10, new Random(1));
		 System.out.println("*****Estimated Accuracy: "
		 + Double.toString(eval2.pctCorrect()));
		 System.out.println("*************sigmoid**********");
		 svm.setOptions(sigmoid);
		 eval2.crossValidateModel(svm, trainingData, 10, new Random(1));
		 System.out.println("*****Estimated Accuracy: "
		 + Double.toString(eval2.pctCorrect()));


		System.out.println("bitti");
	}

}
