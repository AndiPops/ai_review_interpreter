package test;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.Attribute;
import weka.core.DenseInstance;

import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.StringReader;

public class TestModel {
	public static void main(String[] args) {
		try {
			// Load the saved Naive Bayes model
			Classifier classifier = (Classifier) SerializationHelper.read("naiveBayesModel.model");
			System.out.println("Model loaded successfully!");

			// Define the attributes for the instance (review text and sentiment)
			ArrayList<Attribute> attributes = new ArrayList<>();
			attributes.add(new Attribute("text", true));  // "text" is a String attribute for the review
			ArrayList<String> classValues = new ArrayList<>();
			classValues.add("positive");
			classValues.add("negative");
			attributes.add(new Attribute("sentiment", classValues));  // "sentiment" class attribute

			// Create the empty Instances object (dataset)
			Instances dataset = new Instances("MovieReviews", attributes, 0);
			dataset.setClassIndex(1); // Set sentiment as the class attribute

			// Define and test a positive review
			String customReview = "This movie was absolutely fantastic, I loved it!";  // Custom positive review
			System.out.println("Custom review: " + customReview);
			testReview(customReview, classifier, dataset);

			// Define and test a negative review
			customReview = "I hated this movie. It was terrible!";  // Custom negative review
			System.out.println("\nCustom review: " + customReview);
			testReview(customReview, classifier, dataset);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void testReview(String customReview, Classifier classifier, Instances dataset) {
		try {
			// Create a new instance (row) for the dataset and assign it to the dataset
			Instance instance = new DenseInstance(2);  // Two attributes: text and sentiment
			instance.setDataset(dataset);  // Set the dataset for this instance
			instance.setValue(0, customReview);  // Set the review text

			// Apply the StringToWordVector filter (the same one used during training)
			StringToWordVector filter = new StringToWordVector();
			filter.setInputFormat(dataset);
			Instances filteredDataset = Filter.useFilter(dataset, filter);

			// Set the text of the instance to be processed by the filter
			filteredDataset.add(instance);
			filteredDataset = Filter.useFilter(filteredDataset, filter);

			// Print out the filtered dataset to verify
			System.out.println("Filtered Dataset: ");
			System.out.println(filteredDataset);

			// Get the prediction
			double classIndex = classifier.classifyInstance(filteredDataset.instance(0));
			String predictedClass = filteredDataset.classAttribute().value((int) classIndex);

			// Print out the prediction results
			System.out.println("Class index: " + classIndex);
			System.out.println("Predicted sentiment: " + predictedClass);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}