package edu.scranton.cs.se584.wordscore;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.reduce.LongSumReducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

/**
 * A MapReduce job that performs basic sentiment analysis using Yelp reviews.
 * <p>
 * For each word that appears in the text of a review, the job computes a score
 * summarizing the "positivity" or "negativity" of that word.
 * <p>
 * The score for a word is computed by applying the following rules:
 * <ul>
 * <li>Each occurrence in a 5-star review increases the word's score by 2.
 * <li>Each occurrence in a 4-star review increases the word's score by 1.
 * <li>3-star reviews have no impact on word score.
 * <li>Each occurrence in a 2-star review decreases the word's score by 1.
 * <li>Each occurrence in a 1-star review decreases the word's score by 2.
 * </ul>
 * <p>
 * The final score for a word is the net effect of applying the above rules to
 * every review in the input dataset. The idea is that more positive words will
 * appear more prevalently in the positive (5-star and 4-star) reviews and more
 * negative words more prevalently in the negative (2-star and 1-star) reviews.
 * More neutral words should appear in both positive and negative reviews,
 * effectively "canceling out".
 * <p>
 * The output of the job is a single, tab-delimited text file in the specified
 * output directory that contains each word and its score in descending order
 * by score. Words with equal score may appear in any order.
 * <p>
 * TODO(student) Document any improvements that you have made to the algorithm.
 *
 * @author Daniel M. Jackowitz
 * @author Cuong D. Nguyen
 */
public class ComputeWordScore extends Configured implements Tool {
  // You don't need to worry yet about Configured and Tool. We'll discuss them
  // in lecture, but just know that they add some extra CLI parsing logic for
  // framework-level configuration and then eventually call run(String[]) with
  // the remaining, non-framework arguments left for your code to handle.
  // For now, just think of run(String[]) as the standard Java main(String[]).
  // TODO(student) Remove the preceding comment prior to submission.

  /**
   * Produces (.stars, 1), where ".stars" is the value of the JSON field named "stars".
   */
  public static class ComputeWordScoreMapper extends Mapper<LongWritable, Text, Text, LongWritable> {

    // For constant values you can create and share a single static Writable.
    private static final LongWritable PLUS_TWO = new LongWritable(2);
    private static final LongWritable PLUS_ONE = new LongWritable(1);
    private static final LongWritable ZERO = new LongWritable(0);    
    private static final LongWritable MINUS_ONE = new LongWritable(-1);
    private static final LongWritable MINUS_TWO = new LongWritable(-2);

    // Writables in Hadoop can and should be reused across calls to map().
    // Non-constant Writables must not be shared across Mapper instances.
    // private final IntWritable stars = new IntWritable();
    private final Text word = new Text();

    // Each Mapper in MapReduce is only ever called from a single thread.
    private final JSONParser parser = new JSONParser();

    // Helper to encapsulate exception conversion and casting.
    // This is just regular Java JSON parsing, nothing MapReduce-specific.
    private JSONObject parse(final String line) {
      try {
        return (JSONObject) parser.parse(line);
      } catch (ParseException exception) {
        throw new RuntimeException(exception);
      }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      final JSONObject review = parse(value.toString());
      // Update the value of the Writable and write (.stars, 1) to the Context.
      int scoreModifier = ((Number) review.get("stars")).intValue();

      // References: WordCount.java Hadoop Example
      StringTokenizer itr = new StringTokenizer(review.get("text").toString());
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        if (scoreModifier==5) {
          context.write(word, PLUS_TWO);
        } else if (scoreModifier==4) {
          context.write(word, PLUS_ONE);
        } else if (scoreModifier == 3) {
          context.write(word, ZERO);
        } else if (scoreModifier == 2) {
          context.write(word, MINUS_ONE);
        } else if (scoreModifier == 1) {
          context.write(word, MINUS_TWO);
        }
      }
    }
  }

  @Override
  public int run(String[] arguments) throws Exception {
    if (arguments.length < 2) {
      System.err.println("Usage: <input> <output>");
      return 2; // Return 2 to signal a problem with the arguments.
    }

    // This is the second part of the Configured/Tool "magic" mentioned above.
    // You're definitely going to need a Configuration object and you'll want
    // to use the one from getConf() rather than creating your own in order to
    // benefit from the functionality added by Configured/Tool.
    // TODO(student) Remove the preceding comment prior to submission.
    final Configuration conf = getConf();
    final Job job = Job.getInstance(conf, "Compute Word Score");

    // Hadoop needs to know which JAR file contains the job code.
    // This JAR will be uploaded to the cluster when the job is submitted.
    job.setJarByClass(ComputeWordScore.class);

    // We're using files for both input and output, taken from the arguments.
    job.setInputFormatClass(TextInputFormat.class); // This is the default.
    FileInputFormat.addInputPath(job, new Path(arguments[0]));
    job.setOutputValueClass(TextOutputFormat.class); // This is the default.
    FileOutputFormat.setOutputPath(job, new Path(arguments[1]));

    // Register the Mapper and Reducer implementations with the Hadoop
    // framework. Note that unlike in our SimpleMapReduce these are not
    // instances, but the Class objects. This is because Hadoop needs not
    // just a single instance of each, but the ability to construct as many
    // instances of Mappers and Reducers as it requires, one for each task.
    job.setMapperClass(ComputeWordScoreMapper.class);
    job.setReducerClass(LongSumReducer.class);

    // These are required so Hadoop knows how to (de)serialize the output.
    // If it seems redundant, research "type erasure" with Java generics.
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(LongWritable.class);

    // Finally, submit the job and wait for it to complete. Everything
    // prior was just setup, this actually triggers real computation.
    // When using ToolRunner it's expected to return an integer value
    // to be used as the process return code (i.e. 0 for success).
    return job.waitForCompletion(true) ? 0 : 1;
  }

  public static void main(String[] arguments) throws Exception {
    // ToolRunner is a common Hadoop utility that wraps argument parsing with
    // some helpers to populate a Configuration. You can then later access this
    // populated Configuration using getConf().
    // TODO(student) Remove the preceding comment prior to submission.
    System.exit(ToolRunner.run(new ComputeWordScore(), arguments));
  }
}
