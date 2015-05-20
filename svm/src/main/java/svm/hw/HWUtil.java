package svm.hw;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class HWUtil {

	public static Instances writeFileToDataSet() throws FileNotFoundException,
			Exception {
		String filePath = "src/main/resources/optdigits01.txt";
		Scanner digitDataFile = new Scanner(new File(filePath));

		double[][] featureList = new double[360][65];
		int i = 0;
		while (digitDataFile.hasNextLine()) {
			String line = digitDataFile.nextLine();

			Scanner scanner = new Scanner(line);
			scanner.useDelimiter("	");
			int j = 0;
			while (scanner.hasNextInt()) {
				featureList[i][j] = (scanner.nextInt());
				j++;
			}
			scanner.close();
			i++;
		}
		FastVector attInfo = new FastVector(65);
		for (int counter = 0; counter < 65; counter++) {
			Attribute attribute = new Attribute("Att No :"
					+ String.valueOf(counter));
			attInfo.addElement(attribute);
		}
		Instances dataset = new Instances("trainingset", attInfo, 360);
		for (int row = 0; row < 360; row++) {
			Instance instance = new Instance(65);
			for (int column = 0; column < 65; column++) {
				instance.setValue(column, featureList[row][column]);
			}
			dataset.add(instance);

		}
		Attribute attribute = dataset.attribute(0);
		System.out.println(attribute);

		dataset.setClassIndex(dataset.numAttributes() - 1);
		NumericToNominal convert = new NumericToNominal();
		int[] classNumeric = new int[1];
		classNumeric[0] = 64;
		convert.setAttributeIndicesArray(classNumeric);
		convert.setInputFormat(dataset);
		Instances trainingData = Filter.useFilter(dataset, convert);
		trainingData.setClassIndex(trainingData.numAttributes() - 1);
		System.out.println(trainingData.classAttribute());
		return trainingData;
	}

	public static void writeToArff(Instances trainingData) throws IOException {
		Instances dataSet = trainingData;
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);
		saver.setFile(new File("src/main/resources/optdigits01.arff"));
		saver.setDestination(new File("src/main/resources/optdigits01.arff")); // **not**
																				// necessary
																				// in
																				// 3.5.4
																				// and
																				// later
		saver.writeBatch();
	}

}
