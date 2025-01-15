package preprocessing;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Preprocessor {
	private static final List<String> STOPWORDS = Arrays.asList("a", "an", "the", "and", "or", "but");

	public static String cleanText(String input) {
		// lowercase
		String text = input.toLowerCase();

		// remove punctuation
		text = text.replaceAll("[^a-zA-Z\\s]", "");

		// remove stopwords
		List<String> words = Arrays.asList(text.split("\\s+"));
		words = words.stream()
				.filter(word -> !STOPWORDS.contains(word))
				.collect(Collectors.toList());

		return String.join(" ", words);
	}
}