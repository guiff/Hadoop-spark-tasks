package task2pkg;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Set;
import java.util.List;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;



public class task2 {

	//Mapper1: tokenize file
	public static class Mapper1 extends Mapper<Object, Text, Text, IntWritable> {
			
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
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        	String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();  //Get the file name
        	value.set(value.toString().toLowerCase()); //Make all the words lower case
            StringTokenizer itr = new StringTokenizer(value.toString());
            
            while (itr.hasMoreTokens()) {
            	word.set(itr.nextToken());
            	if(stopwords.contains(word.toString())) {
            		continue; //Don't take stopwords into account
            	}
            	String[] words = word.toString().split("\\p{P}"); //Split using punctuation
            	for (String w: words) {
            		if (w.isEmpty()) {
            			continue; //Don't take empty strings into account
            		}
            		word.set(w + "@" + fileName);
            		context.write(word, one);
            	}
            }
        }
	}
	
	//Mapper2: output a pair (word, filename = frequency)
	public static class Mapper2 extends Mapper<Object, Text, Text, Text> {

		private Text word = new Text();
		private Text fileNameAndFrequency = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException { 
			StringTokenizer itr = new StringTokenizer(value.toString());
			while(itr.hasMoreTokens()) {
				String[] split = itr.nextToken().split("@");
				word.set(split[0]);
				fileNameAndFrequency.set(split[1] + "=" + itr.nextToken());
				context.write(word, fileNameAndFrequency);
			}
		}
	}
	
	//Mapper3: output a pair (file name, word = tfidf)
	public static class Mapper3 extends Mapper<Object, Text, Text, Text> {

		private Text fileName = new Text();
		private Text wordAndTfidf = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			StringTokenizer itr = new StringTokenizer(value.toString());
			while(itr.hasMoreTokens()) {
				String[] split = itr.nextToken().split("@");
				fileName.set(split[1]);
				wordAndTfidf.set(split[0] + "=" + itr.nextToken());
				context.write(fileName, wordAndTfidf);
			}
		}
	}
	
	//Mapper4: read the query in the setup phase and output a pair (file name, word = normalized tf-idf)
	public static class Mapper4 extends Mapper<Object, Text, Text, Text> {
			
        private Text fileName = new Text();
        private Text wordAndNormalizedTfidf = new Text();
		private Set<String> query = new HashSet<String>();      
	    
		@Override
		//Load query file from HDFS and parse content into a set of words
	    protected void setup(Context context) throws IOException, InterruptedException {
	        Configuration conf = context.getConfiguration();
	        try {
	        	Path path = new Path("hdfs://localhost:9000/data/inout/query.txt");
	        	FileSystem fs = path.getFileSystem(conf);
	            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
	            String word = null;
	            while ((word = br.readLine())!= null) {
	            	query.add(word);
	            }
	        } catch (IOException e) {
	        	e.printStackTrace();
	        }
	    }
        
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            StringTokenizer itr = new StringTokenizer(value.toString());
            
            while (itr.hasMoreTokens()) {
				String[] split = itr.nextToken().split("@");
				
            	if(! query.contains(split[0])) {
            		continue; //Keep only the query words
            	}
				fileName.set(split[1]);
				wordAndNormalizedTfidf.set(split[0] + "=" + itr.nextToken());
				context.write(fileName, wordAndNormalizedTfidf);
            }
        }
	}
	
	//Mapper5: return the opposite value of the relevance as key, and the fileName as value
	public static class Mapper5 extends Mapper<Object, Text, DoubleWritable, Text> {

		private DoubleWritable relevance = new DoubleWritable();
		private Text fileName = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException { 
			StringTokenizer itr = new StringTokenizer(value.toString());
			while(itr.hasMoreTokens()) {
				fileName.set(itr.nextToken());
				// We take the opposite value of the relevance to have descending order sorting
				relevance.set(- Double.parseDouble(itr.nextToken()));
				context.write(relevance, fileName);
			}
		}
	}
	
    //Reducer1: calculate frequency of every word in every file
    public static class Reducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
        
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
    
    //Reducer2: calculate tf-idf of every word in every document
    public static class Reducer2 extends Reducer<Text, Text, Text, DoubleWritable> {
        
    	private Text word = new Text();
    	private DoubleWritable tfidf = new DoubleWritable();
 
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        	
        	//Get the total number of documents
        	Configuration conf = context.getConfiguration();
        	Path path = new Path(conf.get("inputPath"));
        	FileSystem fs = path.getFileSystem(conf);
        	long totalDocumentsNumber = fs.getContentSummary(path).getFileCount();
        	
        	double documentsContainingWord = 0;
        	List<String> valuesList = new ArrayList<String>();
        	
            for (Text val : values) {
            	valuesList.add(val.toString()); //Store the values in a list because it is impossible to iterate twice over them
            	documentsContainingWord++;
            }
            for (String val : valuesList) { //Iterate over the created list to calculate the tf-idf
            	String[] split = val.split("=");
            	word.set(key + "@" + split[0]);
                tfidf.set((1 + Math.log(Integer.parseInt(split[1]))) * Math.log(totalDocumentsNumber / documentsContainingWord));
                context.write(word, tfidf);
            }
        }
    }
    
    //Reducer3: calculate normalized tf-idf of every word in every document
    public static class Reducer3 extends Reducer<Text, Text, Text, DoubleWritable> {
        
    	private Text word = new Text();
    	private DoubleWritable normTfidf = new DoubleWritable();
 
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

        	double divisor = 0;
        	List<String> valuesList = new ArrayList<String>();
        	
            for (Text val : values) {
            	valuesList.add(val.toString()); //Store the values in a list because it is impossible to iterate twice over them
            	divisor += Math.pow(Double.parseDouble(val.toString().split("=")[1]), 2);
            }
            
            for (String val : valuesList) { //Iterate over the created list to calculate the normalized tf-idf
            	String[] split = val.split("=");
            	word.set(split[0] + "@" + key);
                normTfidf.set(Double.parseDouble(split[1]) / Math.sqrt(divisor));
                context.write(word, normTfidf);
            }
        }
    }
    
    //Reducer4: calculate relevance of every document w.r.t. a query
    public static class Reducer4 extends Reducer<Text, Text, Text, DoubleWritable> {
        
    	private Text fileName = new Text();
    	private DoubleWritable totalNormTfidf = new DoubleWritable();
 
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

        	double sum = 0;
        	
            for (Text val : values) {
            	sum += Double.parseDouble(val.toString().split("=")[1]);
            }
            
        	fileName.set(key);
            totalNormTfidf.set(sum);
            context.write(fileName, totalNormTfidf);
        }
    }
    
	//Reducer5: output the relevance as the key and the fileName as the value
	public static class Reducer5 extends Reducer<DoubleWritable, Text, DoubleWritable, Text> {

		public void reduce(DoubleWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			for (Text val: values) {
				key.set(-key.get()); //We take again the opposite value to get the original relevance back
				context.write(key, val);
			}
		}
	}
    
    
    
    //main function///////////////////////////////////////////////////////////////////////////////////
	    
    public static void main(String[] args) throws Exception {
        
    	Configuration conf = new Configuration();
    	
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 2) {
            System.err.println("Usage: <inputFolder> <outputFolder>");
            System.exit(2);
        }
	    
        conf.set("inputPath", otherArgs[0]); //Needed in Reducer2 to get the number of documents processed, to calculate the tf-idf
        
        
	    // Stage 1: compute the frequency of every word in a document
        Job job1 = new Job(conf, "Frequency of every word in a document");
        job1.setJarByClass(task2.class);
        job1.setMapperClass(Mapper1.class);
        job1.setCombinerClass(Reducer1.class);
        job1.setReducerClass(Reducer1.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        job1.setNumReduceTasks(1);
        
        FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job1, new Path("hdfs://localhost:9000/data/tempTask2/stage1"));
        job1.waitForCompletion(true);
        
		// Stage 2: compute tf-idf of every word w.r.t. a document
		Job job2 = new Job(conf, "Compute tf-idf");
		job2.setJarByClass(task2.class);
		job2.setMapperClass(Mapper2.class);
		job2.setReducerClass(Reducer2.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(DoubleWritable.class);
		job2.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job2, new Path("hdfs://localhost:9000/data/tempTask2/stage1"));
		FileOutputFormat.setOutputPath(job2, new Path("hdfs://localhost:9000/data/tempTask2/stage2"));
		job2.waitForCompletion(true);
		
		// Stage 3: compute normalized tf-idf of every word w.r.t. a document
		Job job3 = new Job(conf, "Compute normalized tf-idf");
		job3.setJarByClass(task2.class);
		job3.setMapperClass(Mapper3.class);
		job3.setReducerClass(Reducer3.class);
		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(Text.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(DoubleWritable.class);
		job3.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job3, new Path("hdfs://localhost:9000/data/tempTask2/stage2"));
		FileOutputFormat.setOutputPath(job3, new Path("hdfs://localhost:9000/data/tempTask2/stage3"));
		job3.waitForCompletion(true);
		
		// Stage 4: compute the relevance of every document w.r.t. a query
		Job job4 = new Job(conf, "Compute relevance w.r.t. a query");
		job4.setJarByClass(task2.class);
		job4.setMapperClass(Mapper4.class);
		job4.setReducerClass(Reducer4.class);
		job4.setMapOutputKeyClass(Text.class);
		job4.setMapOutputValueClass(Text.class);
		job4.setOutputKeyClass(Text.class);
		job4.setOutputValueClass(DoubleWritable.class);
		job4.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job4, new Path("hdfs://localhost:9000/data/tempTask2/stage3"));
		FileOutputFormat.setOutputPath(job4, new Path("hdfs://localhost:9000/data/tempTask2/stage4"));
		job4.waitForCompletion(true);
		
		// Stage 5: sort the documents by their relevance to the query
		Job job5 = new Job(conf, "Sort the documents");
		job5.setJarByClass(task2.class);
		job5.setMapperClass(Mapper5.class);
		job5.setReducerClass(Reducer5.class);
		job5.setMapOutputKeyClass(DoubleWritable.class);
		job5.setMapOutputValueClass(Text.class);
		job5.setOutputKeyClass(DoubleWritable.class);
		job5.setOutputValueClass(Text.class);
		job5.setNumReduceTasks(1);
		
		FileInputFormat.addInputPath(job5, new Path("hdfs://localhost:9000/data/tempTask2/stage4"));
		FileOutputFormat.setOutputPath(job5, new Path(otherArgs[1]));
		System.exit(job5.waitForCompletion(true) ? 0 : 1);
    }
}
