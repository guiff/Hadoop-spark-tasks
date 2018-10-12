import java.io.File
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Assignment1 {
	def main(args: Array[String]) {
		val inputFolder = "/home/hadoop/Downloads/InputFiles"
		val totalDocumentsNumber = new File(inputFolder).listFiles.size.toDouble
		val conf = new SparkConf().setAppName("Top k similar documents").setMaster("local[2]")
		val sc = new SparkContext(conf)

		// Get the query words
		val queryFile = sc.textFile("file:///home/hadoop/Downloads/query.txt").cache()
		val queryWords = queryFile.flatMap(x => x.split(" ")).map(_.trim)		
		val broadcastQueryWords = sc.broadcast(queryWords.collect.toSet)

		
		// Stage 1: compute frequency of every word in a document
	
		// Get the stopwords
		val stopWordsFile = sc.textFile("file:///home/hadoop/Downloads/stopwords.txt").cache()
		val stopWords = stopWordsFile.flatMap(x => x.split(" ")).map(_.trim)
		val broadcastStopWords = sc.broadcast(stopWords.collect.toSet)
		
		var counts = sc.emptyRDD[(String, Int)]

		for (file <- new File(inputFolder).listFiles) {
			val fileName = file.getName()
			val inputFile = sc.textFile("file:/" + "/" + inputFolder + "/" + fileName).cache()	
			var words = inputFile.flatMap(line => line.toLowerCase.split(" "))		
			words = words.filter(!broadcastStopWords.value.contains(_)) // Remove the stopwords
			words = words.flatMap(word => word.split("[\\p{Punct}\\s]+")) // Split using punctuation
			words = words.filter(!broadcastStopWords.value.contains(_)) // Remove again possible remaining stopwords after punctuation splitting
			val fileCounts = words.map(word => (word + "@" + fileName, 1)).reduceByKey(_ + _)
			counts ++= fileCounts
		}
		
		
		// Stage 2: compute tf-idf of every word wrt to a document

		val map2_1 = counts.map{case (key, value) => (key.split("@")(0), key.split("@")(1) + "=" + value)}
		val documentFrequency = map2_1.countByKey
		val map2_2 = map2_1.map{case (key, value) => (key + "@" + value.split("=")(0), (1 + math.log(value.split("=")(1).toDouble)) * math.log(totalDocumentsNumber / documentFrequency(key).toDouble))} // Compute tf-idf
		

		// Stage 3: compute normalized tf-idf
		
		val map3_1 = map2_2.map{case (key, value) => (key.split("@")(1), key.split("@")(0) + "=" + value)}
		val divisor = map3_1.reduceByKey((value1, value2) => "result=" + (value1.split("=")(1).toDouble + math.pow(value2.split("=")(1).toDouble, 2))).collectAsMap()
		val map3_2 = map3_1.map{case (key, value) => (value.split("=")(0) + "@" + key, value.split("=")(1).toDouble / math.sqrt(divisor(key).split("=")(1).toDouble))} // Compute normalized tf-idf


		// Stage 4: compute the relevance of every document wrt a query

		val filter = map3_2.filter{case (key, value) => broadcastQueryWords.value.contains(key.split("@")(0))} // Keep only the query words
		val map4_1 = filter.map{case (key, value) => (key.split("@")(1), key.split("@")(0) + "=" + value)}
		val reduce4 = map4_1.reduceByKey((value1, value2) => "result=" + (value1.split("=")(1).toDouble + value2.split("=")(1).toDouble)) // Sum the normalized tf-idf
		val map4_2 = reduce4.map{case (key, value) => (key, value.split("=")(1).toDouble)}
		

		// Stage 5: sort documents by their relevance to the query

		val sortedResults = map4_2.map(item => item.swap).sortByKey(false)
		sortedResults.repartition(1).saveAsTextFile("file:///home/hadoop/Downloads/output/") // Save the result as a text file
	}
}
