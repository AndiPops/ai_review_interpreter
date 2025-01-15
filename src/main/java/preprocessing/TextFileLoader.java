package preprocessing;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;

public class TextFileLoader {
	public static void main(String[] args) {
		try {
			// Define file paths for training data (adjust the path as necessary)
			File positiveFolder = new File("data/aclImdb/train/pos");
			File negativeFolder = new File("data/aclImdb/train/neg");

			// Define attributes: text attribute and sentiment (positive/negative) class attribute
			ArrayList<Attribute> attributes = new ArrayList<>();
			attributes.add(new Attribute("text", true));  // "text" is a String attribute for the reviews
			ArrayList<String> classValues = new ArrayList<>();
			classValues.add("positive");
			classValues.add("negative");
			attributes.add(new Attribute("sentiment", classValues));  // Class attribute for sentiment

			// Create an empty Instances object (dataset)
			Instances dataset = new Instances("MovieReviews", attributes, 0);
			dataset.setClassIndex(1); // Set sentiment as the class attribute

			// Load positive reviews
			loadReviewsFromFolder(positiveFolder, dataset, "positive");

			// Load negative reviews
			loadReviewsFromFolder(negativeFolder, dataset, "negative");

			// Apply StringToWordVector filter to preprocess the text data
			StringToWordVector filter = new StringToWordVector();
			filter.setInputFormat(dataset);  // Set the input format for the filter
			Instances filteredDataset = Filter.useFilter(dataset, filter);  // Apply the filter

			// Train a Naive Bayes classifier
			Classifier classifier = new NaiveBayes();
			classifier.buildClassifier(filteredDataset);

			// Evaluate the classifier using cross-validation
			Evaluation evaluation = new Evaluation(filteredDataset);
			evaluation.crossValidateModel(classifier, filteredDataset, 10, new java.util.Random(1));  // 10-fold cross-validation

			// Print evaluation results
			System.out.println("Evaluation results:");
			System.out.println(evaluation.toSummaryString());

			// Save the classifier to a file
			try {
				SerializationHelper.write("naiveBayesModel.model", classifier);
				System.out.println("Model saved!");
			} catch (Exception e) {
				System.out.println("Error saving model: " + e.getMessage());
			}

		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	// Method to load reviews from a folder and add them to the dataset
	private static void loadReviewsFromFolder(File folder, Instances dataset, String sentiment) throws IOException {
		// Iterate through each file in the folder
		for (File file : folder.listFiles()) {
			if (file.isFile()) {
				// Read the content of the review text file
				String reviewText = readFile(file);

				// Create a new instance (row) for the dataset
				Instance instance = new DenseInstance(dataset.numAttributes());  // Create instance based on the dataset's attribute count
				instance.setDataset(dataset);  // Link instance to the dataset
				instance.setValue(0, reviewText);  // Set the review text
				instance.setClassValue(sentiment);  // Set the sentiment class (positive/negative)

				// Add the instance to the dataset
				dataset.add(instance);
			}
		}
	}

	// Method to read the content of a file and return it as a string
	private static String readFile(File file) throws IOException {
		StringBuilder stringBuilder = new StringBuilder();
		BufferedReader reader = new BufferedReader(new FileReader(file));

		String line;
		while ((line = reader.readLine()) != null) {
			stringBuilder.append(line).append(" ");
		}
		reader.close();
		return stringBuilder.toString();
	}
}