import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

//----------------------------------------------------------------------------------------------------------------------

public class Kmeans {

    final public static ArrayList<ArrayList<Double>> matrix = new ArrayList<>(); //Matrix of all the data-points
    public static int numberOfClusters;
    public static ArrayList<ArrayList<Double>> centroidList = new ArrayList<>();   //List containing the centroids
    public static ArrayList<Integer> mainList = new ArrayList<>();   //List containing clusterId assigned to each data point
    public static ArrayList<Integer> trueValues = new ArrayList<>();    //List containing ground truth of each data point

    /*
    The method readFile() populates the matrix with data points.
    It also initializes mainList with -1 and fills up the trueValues list.
     */

    //------------------------------------------------------------------------------------------------------------------

    public static void readFile(String filePath) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(
                new FileReader(filePath));
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            String[] array = line.split("\t");
            ArrayList<Double> list = new ArrayList<>();
            for (int i = 2; i < array.length; i++) {
                list.add(Double.parseDouble(array[i]));
            }
            matrix.add(list);     //Matrix contains points at each row
            mainList.add(-1);
            trueValues.add(Integer.parseInt(array[1]));
        }

//        Random random = new Random();
//        numberOfClusters = 5;
//        while(centroidList.size()<numberOfClusters) {    //Centroid list contains the centroids
//            int randomInt = random.nextInt(matrix.size());
//            if(!centroidList.contains(matrix.get(randomInt)))
//                centroidList.add(matrix.get(randomInt));
//        }

        centroidList.add(matrix.get(5));
        centroidList.add(matrix.get(25));
        centroidList.add(matrix.get(32));
        centroidList.add(matrix.get(100));
        centroidList.add(matrix.get(132));
    }

    //------------------------------------------------------------------------------------------------------------------

    public static class MapperClass extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] array = line.split("\t");
            ArrayList<Double> point = new ArrayList<>();
            for(int i=2; i<array.length; i++) {
                point.add(Double.parseDouble(array[i]));
            }
            int pointInt = Integer.parseInt(array[0]) - 1;  //PointInt is the index of the data-point
            Double minDist = Double.MAX_VALUE;
            int finalCentroid = 0;                          //Final centroid will store index of centroid from centroidList which the point is closest to
            for(int i=0; i<centroidList.size(); i++) {    //Here it calculates the centroid which the point is closest to
                ArrayList<Double> centroid = centroidList.get(i);
                double sum = 0;
                for(int j=0; j<centroid.size(); j++) {
                    sum += Math.pow(centroid.get(j)-point.get(j), 2);
                }
                double distance = Math.sqrt(sum);
                if(distance < minDist){
                    minDist = distance;
                    finalCentroid = i;
                }

            }
            IntWritable pointId = new IntWritable(pointInt);
            IntWritable clusterId = new IntWritable(finalCentroid);
            context.write(clusterId, pointId);  //We're returning centroid index and the index of the point associated with that centroid
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public static class ReducerClass extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int clusterId = key.get();
            int count = 0;
            ArrayList<ArrayList<Double>> clusterPoints = new ArrayList<>();
            for(IntWritable value : values) {
                int rowId = value.get();
                mainList.set(rowId, clusterId);     //Update values in mainList where index is the point index and value is the assigned clusterId.
                clusterPoints.add(matrix.get(rowId));
                count++;
            }
            ArrayList<Double> newCentroid = new ArrayList<>();
            for(int i=0; i<clusterPoints.get(0).size(); i++) { //Recompute new centroid
                double sum = 0;
                for(int j=0; j<clusterPoints.size(); j++) {
                    sum += clusterPoints.get(j).get(i);
                }
                sum /= Double.parseDouble(count+"");
                newCentroid.add(sum);
            }
            centroidList.set(clusterId, newCentroid); //Replace old centroid with new centroid in centroidList
            IntWritable numPointsAssignedToCluster = new IntWritable(count);
            context.write(key, numPointsAssignedToCluster);
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        readFile(args[0]);
        Configuration conf = new Configuration();
        Path inPath = new Path(args[0]);
        Path outPath =  null;
        int j = 1;
        while (true){
            outPath = new Path("/home/ashwin/output/"+args[1]+j);
            j++;
            Job job = new Job(conf, "word count");
            job.setJarByClass(Kmeans.class);
            job.setMapperClass(MapperClass.class);
            job.setReducerClass(ReducerClass.class);
            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(IntWritable.class);
            job.setOutputKeyClass(IntWritable.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.addInputPath(job, inPath);
            FileOutputFormat.setOutputPath(job, outPath);
            ArrayList tempList = new ArrayList(mainList);
            job.waitForCompletion(true);
            if(tempList.equals(mainList)) //Iterations carry on until no points move from one cluster to another
                break;
        }
        calculateEfficiency();
    }

    //------------------------------------------------------------------------------------------------------------------

    public static void calculateEfficiency() {
        int[][] groundTruth = new int[matrix.size()][matrix.size()];
        int[][] clusterTruth = new int[matrix.size()][matrix.size()];
        for(int i=0; i<matrix.size(); i++) {
            for(int j=i; j<matrix.size(); j++) {
                if(trueValues.get(i) == trueValues.get(j)) {
                    groundTruth[i][j] = 1;
                    groundTruth[j][i] = 1;
                }
                else {
                    groundTruth[i][j] = 0;
                    groundTruth[j][i] = 0;
                }
                if(mainList.get(i) == mainList.get(j)) {
                    clusterTruth[i][j] = 1;
                    clusterTruth[j][i] = 1;
                } else {
                    clusterTruth[i][j] = 0;
                    clusterTruth[j][i] = 0;
                }
            }
        }

        double m00 = 0, m01 = 0, m10 = 0, m11 = 0;

        for(int i=0; i<matrix.size(); i++) {
            for(int j=0; j<matrix.size(); j++) {
                if(groundTruth[i][j] == 0 && clusterTruth[i][j] == 0)
                    m00++;
                else if(groundTruth[i][j] == 0 && clusterTruth[i][j] == 1)
                    m01++;
                else if(groundTruth[i][j] == 1 && clusterTruth[i][j] == 0)
                    m10++;
                else if(groundTruth[i][j] == 1 && clusterTruth[i][j] == 1)
                    m11++;
            }
        }

        double randIndex, jCoeff;
        randIndex = (m00 + m11)/(m00 + m01 + m10 + m11);
        jCoeff = (m11)/(m11 + m01 + m10);
        System.out.println("Rand: "+randIndex+" ");
        System.out.println("Jaccard Coefficient: "+jCoeff);
    }
}

//----------------------------------------------------------------------------------------------------------------------