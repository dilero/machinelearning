package svm.hw;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class ClassifierExample {

	public static void main(String[] args) throws Exception {

		Instances trainingData = HWUtil.writeFileToDataSet();

		// HWUtil.writeToArff(trainingData);

		svm(trainingData);

		// mlp(trainingData);

	}

	private static void mlp(Instances trainingData) throws Exception {
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		String[] hidden = weka.core.Utils.splitOptions("-H 2");
		mlp.setOptions(hidden);
		Evaluation eval2 = new Evaluation(trainingData);
		eval2.crossValidateModel(mlp, trainingData, 10, new Random(1));
		System.out.println("*****Estimated Accuracy: "
				+ Double.toString(eval2.pctCorrect()));
	}

	private static void svm(Instances trainingData) throws Exception {
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
	}

}
