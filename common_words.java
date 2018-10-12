package task1pkg;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Set;
import java.util.Arrays;
import java.util.HashSet;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;



public class task1 {

	//Word count mapper: output each word of the text as a key with a value of 1
	public static class WordCountMapper extends Mapper<Object, Text, Text, IntWritable> {

		private Text word = new Text();
		private final static IntWritable one = new IntWritable(1);
		private Set<String> stopwords = new HashSet<String>();      

		@Override
		//Load stopwords file from HDFS and parse content into a set of words
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			try {
				Path path = new Path("hdfs://localhost:9000/data/inout/stopwords.txt");
				FileSystem fs = path.getFileSystem(conf);
				BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
				String word = null;
				while ((word = br.readLine())!= null) {
					stopwords.add(word);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		//Map function filtering the stop words
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			value.set(value.toString().replaceAll("\\p{P}", "").toLowerCase()); //Get rid of punctuation and make the words lower case
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				if(stopwords.contains(word.toString())) {
					continue; //Don't take stopwords into account
				}
				context.write(word, one);
			}
		}
	}

	//Common words count mapper 1: add "s1" after word count of file 1
	public static class CommonWordsCountMapper1 extends Mapper<Object, Text, Text, Text> {

		private Text word = new Text();
		private Text frequency = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				frequency.set(itr.nextToken()+"_s1");
				context.write(word, frequency);
			}
		}
	}

	//Common words count mapper 2: add "s2" after word count of file 2
	public static class CommonWordsCountMapper2 extends Mapper<Object, Text, Text, Text> {

		private Text word = new Text();
		private Text frequency = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while (itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				frequency.set(itr.nextToken()+"_s2");
				context.write(word, frequency);
			}
		}
	}

	//Sort mapper: return the opposite value of the common word count as a key, with the corresponding word as value
	public static class SortMapper extends Mapper<Object, Text, IntWritable, Text> {

		private IntWritable number = new IntWritable();
		private Text word = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException { 
			StringTokenizer itr = new StringTokenizer(value.toString());
			while(itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				// We take the opposite value of the common word count to have descending order sorting
				number.set(- Integer.parseInt(itr.nextToken()));
				context.write(number, word);
			}
		}
	}

	//Word count reducer: sum the values linked to a word
	public static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	//Common words reducer: get the number of common words
	public static class CommonWordsCountReducer extends Reducer<Text, Text, Text, IntWritable> {

		private IntWritable result = new IntWritable();

		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			int count = 0;
			int frequencyArray[] = new int[]{0,0};
			for (Text val : values) {
				String frequency = val.toString().split("_")[0];
				frequencyArray[count] = Integer.parseInt(frequency);
				count++;
			}

			if (count == 2) { //If the word is present in the two files
				Arrays.sort(frequencyArray); //Keep the smallest value
				result.set(frequencyArray[0]);
				context.write(key, result);
			}
		}
	}

	//Sort reducer: simply output the common word count as the key and the word as the value
	public static class SortReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			for (Text val: values) {
				key.set(-key.get()); //We take again the opposite value to get the original value back
				context.write(key, val);
			}
		}
	}




	///////////////////////////////////////////////////////////////////////////////////
	//main function

	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

		if (otherArgs.length != 3) {
			System.err.println("Usage: <input1> <input2> <output>");
			System.exit(2);
		}


		// Stage 1: word count for file 1
		Job job1 = new Job(conf, "Word count for file 1");
		job1.setJarByClass(task1.class);		
		job1.setMapperClass(WordCountMapper.class);
		job1.setCombinerClass(WordCountReducer.class);
		job1.setReducerClass(WordCountReducer.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(IntWritable.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);
		job1.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job1, new Path("hdfs://localhost:9000/data/tempTask1/wordcount1"));
		job1.waitForCompletion(true);


		// Stage 2: word count for file 2
		Job job2 = new Job(conf, "Word count for file 2");
		job2.setJarByClass(task1.class);
		job2.setMapperClass(WordCountMapper.class);
		job2.setCombinerClass(WordCountReducer.class);
		job2.setReducerClass(WordCountReducer.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(IntWritable.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(IntWritable.class);
		job2.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job2, new Path(otherArgs[1]));
		FileOutputFormat.setOutputPath(job2, new Path("hdfs://localhost:9000/data/tempTask1/wordcount2"));
		job2.waitForCompletion(true);

		//Stage 3: count words in common
		Job job3 = new Job(conf, "Count words in common");
		job3.setJarByClass(task1.class);

		MultipleInputs.addInputPath(job3, new Path("hdfs://localhost:9000/data/tempTask1/wordcount1"), TextInputFormat.class, CommonWordsCountMapper1.class);
		MultipleInputs.addInputPath(job3, new Path("hdfs://localhost:9000/data/tempTask1/wordcount2"), TextInputFormat.class, CommonWordsCountMapper2.class);

		job3.setReducerClass(CommonWordsCountReducer.class);
		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(Text.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(IntWritable.class);
		job3.setNumReduceTasks(1);
		FileOutputFormat.setOutputPath(job3, new Path("hdfs://localhost:9000/data/tempTask1/unsortedCommonWords"));
		job3.waitForCompletion(true);

		//Stage 4: sort in descending order of number of common words
		Job job4 = new Job(conf, "Sort words in descending order");
		job4.setJarByClass(task1.class);
		job4.setMapperClass(SortMapper.class);
		job4.setReducerClass(SortReducer.class);
		job4.setMapOutputKeyClass(IntWritable.class);
		job4.setMapOutputValueClass(Text.class);
		job4.setOutputKeyClass(IntWritable.class);
		job4.setOutputValueClass(Text.class);
		job4.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job4, new Path("hdfs://localhost:9000/data/tempTask1/unsortedCommonWords"));
		FileOutputFormat.setOutputPath(job4, new Path(otherArgs[2]));
		System.exit(job4.waitForCompletion(true) ? 0 : 1);
	}
}

