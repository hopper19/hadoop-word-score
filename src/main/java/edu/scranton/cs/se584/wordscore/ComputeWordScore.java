package edu.scranton.cs.se584.wordscore;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.reduce.IntSumReducer;
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
 *
 * @author Daniel M. Jackowitz
 * @author Cuong D. Nguyen
 */
public class ComputeWordScore extends Configured implements Tool {

  /**
   * Produces (.word, .score_modifier), where:
   *  ".word" is a unique word from the collection of all text reviews.
   *  ".score_modifier" is the modifier value corresponding to the star value of
   *    the review in which the text appears.
   * 
   * To optimize for efficiency, IntWritable is used for output value instead
   * of LongWritable. With roughly seven million reviews, it is unlikely that
   * any word score would reach the bounds of integer range.
   */
  public static class ComputeWordScoreMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    /*
     * Lookup table for score modifiers according to the star value. Shared across
     * calls to map().
     * 
     * Initially, I created five static IntWritable objects and use if statements
     * in the map() method to retrieve the appropriate IntWritable. My next idea
     * was, within map(), to perform the calculation for score, then create the
     * IntWritable. Both ideas are obviously much less efficient than the one below.
     * UPDATE: switch/if may be faster for few entries. Need more testing.
     */
    private static final Map<Integer, IntWritable> SCORE_MODIFIERS = new HashMap<>();
    static {  // Ensure that this code block initializes HashMap, and only execute once.
      SCORE_MODIFIERS.put(5, new IntWritable(2));
      SCORE_MODIFIERS.put(4, new IntWritable(1));
      SCORE_MODIFIERS.put(3, new IntWritable(0));
      SCORE_MODIFIERS.put(2, new IntWritable(-1));
      SCORE_MODIFIERS.put(1, new IntWritable(-2));
    }

    // Non-constant Writable. Must not be shared across Mapper instances.
    private final Text word = new Text();

    // Each Mapper in MapReduce is only ever called from a single thread.
    private final JSONParser parser = new JSONParser();

    // Helper to encapsulate exception conversion and casting.
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
      final IntWritable scoreModifier = SCORE_MODIFIERS.get(((Number) review.get("stars")).intValue());

      // Iterator Reference: WordCount.java Hadoop Example
      StringTokenizer itr = new StringTokenizer(review.get("text").toString());
      while (itr.hasMoreTokens()) {
        word.set(itr.nextToken());
        context.write(word, scoreModifier);
      }
    }
  }

  public static class IntWritableDecreasingComparator extends IntWritable.Comparator {
    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      return super.compare(b2, s2, l2, b1, s1, l1);
    }
  }

  @Override
  public int run(String[] arguments) throws Exception {
    if (arguments.length < 2) {
      System.err.println("Usage: <input> <output>");
      return 2; // Return 2 to signal a problem with the arguments.
    }
    
    final Configuration conf = getConf();

    final Path inputPath = new Path(arguments[0]);
    final Path outputPath = new Path(arguments[1]);

    /*
     * There were a few ideas for the location of the intermediate output directory.
     * Initially, this directory was set to be in the same directory in which the
     * the JAR is executed. The inconvenience here is that each time this same
     * JAR is executed, there are two directories to be manually deleted. Then, I decided to 
     * go with the idea of generating a unique directory name each time. Random
     * UUID was my initial solution. However, in case I need to view the intermediate
     * output, random names prove cumbersome. So, appending the current the current
     * time to the directory name turns out to be a lot more meaningful.Finally,
     * all intermediate directories are created under tmp to delegat the cleanup task to the OS.
     */
    LocalDateTime dateTime = LocalDateTime.now();
    DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
    String dataTimeString = dateTime.format(dateTimeFormatter);
    final Path intermediatePath = new Path("/tmp/wordscore_nguyenc8_" + dataTimeString);

    final Job scorer = Job.getInstance(conf, "Compute Word Score");
    scorer.setJarByClass(ComputeWordScore.class);

    scorer.setInputFormatClass(TextInputFormat.class);
    FileInputFormat.addInputPath(scorer, inputPath);
    /*
     * On SequenceFile(I/O)Format, DMJ,
     * "This format is particularly useful for passing data between multiple jobs…".
     * It eliminated the need to parse this intermediate file in the 2nd job.
     */
    scorer.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(scorer, intermediatePath);

    scorer.setMapperClass(ComputeWordScoreMapper.class);
    scorer.setReducerClass(IntSumReducer.class);
    // Adding a combiner GREATLY reduces I/O access
    scorer.setCombinerClass(IntSumReducer.class);

    scorer.setOutputKeyClass(Text.class);
    scorer.setOutputValueClass(IntWritable.class);


    final Job sorter = Job.getInstance(conf, "Sort Word Score");
    sorter.setJarByClass(ComputeWordScore.class);

    sorter.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(sorter, intermediatePath);
    sorter.setOutputFormatClass(TextOutputFormat.class);
    FileOutputFormat.setOutputPath(sorter, outputPath);

    /*
     * Initially, I wrote my own mapper to swap and sort the scores, but I
     * found out that Hadoop has built-in tools to achieve the same result.
     * 
     * A quick lookup of "descending" and "decreasing" in the Hadoop source code
     * reveals the DecreasingComparator class. However, this class only works with
     * LongWritable. Instead of modifying my program to use LongWrite, in the interest
     * of efficiency, I implemented my own DecreasingComparator for IntWritable.
     * 
     * InverseMapper is one of the Direct Known Subclasses of Mapper in the docs
     */
    sorter.setSortComparatorClass(IntWritableDecreasingComparator.class);
    sorter.setMapperClass(InverseMapper.class);
    // Prevent user from specifying ZERO reduce task
    sorter.setNumReduceTasks(1);

    sorter.setOutputKeyClass(IntWritable.class);
    sorter.setOutputValueClass(Text.class);

    return (scorer.waitForCompletion(true) && sorter.waitForCompletion(true)) ? 0 : 1;
  }

  public static void main(String[] arguments) throws Exception {
    System.exit(ToolRunner.run(new ComputeWordScore(), arguments));
  }
}
