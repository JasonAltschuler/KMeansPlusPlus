/**************************************************************************
 * @author Jason Altschuler
 * 
 * @tags machine learning, computer vision, data mining
 * 
 * PURPOSE: Clusters n-dimensional points.
 * 
 * Algorithm: KMeans++. An improved version of K-Means clustering algorithm.
 * 
 * For full documentation, see readme.txt
 *************************************************************************/

// TODO: plot points for visualization

// TODO: check against MATLAB (measure WCSS)

// TODO: try/catch and run again for empty clusters

// TODO: consistency with terms "clustering" and "iteration" 
// (assignment + update step or 'x' number of those)

// TODO: Give user option to define stopping criteria based on time elapsed

// TODO: Provide other (optional) distance formulas (L1, etc.)

import java.io.IOException;
import java.util.Random;

public class KMeans {

   /***********************************************************************
    * Data structures
    **********************************************************************/
   // user-defined parameters
   private int k;                // number of centroids
   private double[][] points;    // n-dimensional data points. 

   // optional parameters
   private int iterations;       // number of times to repeat the clustering. Choose run with lowest WCSS
   private boolean pp;           // true --> KMeans++. false --> basic random sampling
   private double epsilon;       // stops running when improvement in error < epsilon
   private boolean useEpsilon;   // true  --> stop running when marginal improvement in WCSS < epsilon
                                 // false --> stop running when 0 improvement

   // calculated from dimension of points[][]
   private int m;                // number of data points   (# of pixels for PhenoRipper)  
   private int n;                // number of dimensions    (# of channels for PhenoRipper)

   // output
   private double[][] centroids; // position vectors of centroids                      dim(2): (k) by (number of channels)
   private int[] assignment;     // assigns each point to nearest centroid [0, k-1]    dim(1): (number of pixels)
   private double WCSS;          // within-cluster sum-of-squares. Cost function to minimize

   
   /***********************************************************************
    * Constructors
    **********************************************************************/
   
   /**
    * Empty constructor is private to ensure that clients have to use the 
    * Builder inner class to create a KMeans object.
    */
   private KMeans() {} 

   /**
    * The proper way to construct a KMeans object: from an inner class object.
    * @param builder See inner class named Builder
    */
   private KMeans(Builder builder) {
      // use information from builder
      k = builder.k;
      points = builder.points;
      iterations = builder.iterations;
      pp = builder.pp;
      epsilon = builder.epsilon;
      useEpsilon = builder.useEpsilon;

      // get dimensions to set last 2 fields
      m = points.length;
      n = points[0].length;

      // run KMeans++ clustering algorithm
      run();
   }


   /**
    * Builder class for constructing KMeans objects.
    * 
    * For descriptions of the fields in this (inner) class, see outer class.
    */
   public static class Builder {
      // required
      private final int k;
      private final double[][] points;

      // optional (default values given)
      private int iterations     = 10;
      private boolean pp         = true;
      private double epsilon     = .001;
      private boolean useEpsilon = true;

      /**
       * Sets required parameters.
       */
      public Builder(int k, double[][] points) {
         if (k >= points.length)
            throw new IllegalArgumentException("Required: # of points > # of clusters");
         this.k = k;
         this.points = points;
      }

      /**
       * Sets optional parameter. Default value is 50. 
       */
      public Builder iterations(int iterations) {
         if (iterations < 1) 
            throw new IllegalArgumentException("Required: non-negative number of iterations. Ex: 50");
         this.iterations = iterations;
         return this;
      }

      /**
       * Sets optional parameter. Default value is true.
       */
      public Builder pp(boolean pp) {
         this.pp = pp;
         return this;
      }

      /**
       * Sets optional parameter. Default value is .001.
       */
      public Builder epsilon(double epsilon) {
         if (epsilon < 0.0)
            throw new IllegalArgumentException("Required: non-negative value of epsilon. Ex: .001"); 

         this.epsilon = epsilon;
         return this;
      }

      /**
       * Sets optional parameter. Default value is true.
       */
      public Builder useEpsilon(boolean useEpsilon) {
         this.useEpsilon = useEpsilon;
         return this;
      }

      /**
       * Build a KMeans object
       */
      public KMeans build() {
         return new KMeans(this);
      }
   }


   /***********************************************************************
    * KMeans clustering algorithm
    **********************************************************************/

   /** 
    * Run KMeans algorithm
    */
   private void run() {
      // for choosing the best run
      double bestWCSS = Double.POSITIVE_INFINITY;
      double[][] bestCentroids = new double[0][0];
      int[] bestAssignment = new int[0];

      // run multiple times and then choose the best run
      for (int n = 0; n < iterations; n++) {
         cluster();

         // store info if it was the best run so far
         if (WCSS < bestWCSS) {
            bestWCSS = WCSS;
            bestCentroids = centroids;
            bestAssignment = assignment;
         }
      }

      // keep info from best run
      WCSS = bestWCSS;
      centroids = bestCentroids;
      assignment = bestAssignment;
   }


   /**
    * Perform KMeans clustering algorithm once.
    */
   private void cluster() {
      // continue to re-cluster until marginal gains are small enough
      chooseInitialCentroids();
      WCSS = Double.POSITIVE_INFINITY; 
      double prevWCSS;
      do {  
         assignmentStep();   // assign points to the closest centroids

         updateStep();       // update centroids

         prevWCSS = WCSS;    // check if cost function meets stopping criteria
         calcWCSS();
      } while (!stop(prevWCSS));
   }


   /** 
    * Assigns to each data point the nearest centroid.
    * 
    * TODO: is the following line true?
    * Also used in identification of pixel, block, and superblock types (after running k-means).
    */
   private void assignmentStep() {
      assignment = new int[m];

      double tempDist;
      double minValue;
      int minLocation;

      for (int i = 0; i < m; i++) {
         minLocation = 0;
         minValue = Double.POSITIVE_INFINITY;
         for (int j = 0; j < k; j++) {
            tempDist = EuclideanDistSq(points[i], centroids[j]);
            if (tempDist < minValue) {
               minValue = tempDist;
               minLocation = j;
            }
         }

         assignment[i] = minLocation;
      }
   }


   /** 
    * Updates the centroids.
    */
   private void updateStep() {
      centroids = new double[k][n];
      int[] clustSizes = new int[k];

      // sum points assigned to each cluster
      int assignedClust;
      for (int j = 0; j < m; j++) {
         assignedClust = assignment[j];
         clustSizes[assignedClust]++;
         for (int m = 0; m < n; m++)
            centroids[assignedClust][m] += points[j][m];
      }

      // divide to get averages -> centroids
      for (int c = 0; c < k; c++)
         for (int m = 0; m < n; m++)
            centroids[c][m] /= clustSizes[c];
   }


   /***********************************************************************
    * Choose initial centroids
    **********************************************************************/
   /**
    * Uses either plusplus (KMeans++) or a basic randoms sample to choose initial centroids
    */
   private void chooseInitialCentroids() {
      if (pp)
         plusplus();
      else
         basicRandSample();
   }

   /** 
    * Randomly chooses (without replacement) k data points as initial centroids. 
    */
   private void basicRandSample() {
      centroids = new double[k][n];
      double[][] copy = points;

      Random gen = new Random();

      int rand;
      for (int i = 0; i < k; i++) {
         rand = gen.nextInt(m - i);
         for (int j = 0; j < n; j++) {
            centroids[i][j] = copy[rand][j];       // store chosen centroid
            copy[rand][j] = copy[m - 1 - i][j];    // ensure sampling without replacement
         }
      }
   }

   /** 
    * Randomly chooses (without replacement) k data points as initial centroids using a
    * weighted probability distribution (proportional to D(x)^2 where D(x) is the 
    * distance from a data point to the nearest, already chosen centroid). 
    */
   // TODO: see if some of this code is extraneous (can be deleted)
   private void plusplus() {
      centroids = new double[k][n];    	
      double[] distToClosestCentroid = new double[m];
      double[] weightedDistribution  = new double[m];  // cumulative sum of squared distances

      Random gen = new Random();
      int choose = 0;

      for (int c = 0; c < k; c++) {

         // first centroid: choose any data point
         if (c == 0)
            choose = gen.nextInt(m);

         // after first centroid, use a weighted distribution
         else {

            // check if the most recently added centroid is closer to any of the points than previously added ones
            for (int p = 0; p < m; p++) {
               // gives chosen points 0 probability of being chosen again -> sampling without replacement
               double tempDistance = EuclideanDistSq(points[p], centroids[c - 1]); 

               // base case: if we have only chosen one centroid so far, nothing to compare to
               if (c == 1)
                  distToClosestCentroid[p] = tempDistance;

               else { // c != 1 
                  if (tempDistance < distToClosestCentroid[p])
                     distToClosestCentroid[p] = tempDistance;
               }

               // no need to square because the distance is the square of the euclidean dist
               if (p == 0)
                  weightedDistribution[0] = distToClosestCentroid[0];
               else weightedDistribution[p] = weightedDistribution[p-1] + distToClosestCentroid[p];

            }

            // choose the next centroid
            double rand = gen.nextDouble();
            for (int j = m - 1; j > 0; j--) {
               // TODO: review and try to optimize
               // starts at the largest bin. EDIT: not actually the largest
               if (rand > weightedDistribution[j - 1] / weightedDistribution[m - 1]) { 
                  choose = j; // one bigger than the one above
                  break;
               }
               else // Because of invalid dimension errors, we can't make the forloop go to j2 > -1 when we have (j2-1) in the loop.
                  choose = 0;
            }
         }  

         // store the chosen centroid
         for (int i = 0; i < n; i++)
            centroids[c][i] = points[choose][i];
      }   
   }


   /***********************************************************************
    * Cutoff to stop clustering
    **********************************************************************/    

   /**
    * Calculates whether to stop the run
    * @param prevWCSS error from previous step in the run
    * @return
    */
   private boolean stop(double prevWCSS) {
      if (useEpsilon)
         return epsilonTest(prevWCSS);
      else
         return prevWCSS == WCSS; // TODO: make comment (more exact, but could be much slower)
      // could this take infinite amount of time? double compare...
      // I think not because WCSS is calc in same way as prevWCSS (if data structs don't change)
   }

   /**
    * Signals to stop running KMeans when the marginal improvement in WCSS
    * from the last step is small.
    * @param prevWCSS error from previous step in the run
    * @return
    */
   private boolean epsilonTest(double prevWCSS) {
      return epsilon > 1 - (WCSS / prevWCSS);
   }

   /***********************************************************************
    * Utility functions
    **********************************************************************/
   // TODO: offer other distance functions (maybe in a separate Dist.java class)
   /** 
    * Calculates the squared Euclidean Distance between two points
    * squared because taking square roots could become a bottleneck
    */
   private static double EuclideanDistSq(double[] x, double[] y) {
      if (x.length != y.length)
         throw new IllegalArgumentException("Invalid dimensions.");

      double dist = 0;

      for (int i = 0; i < x.length; i++)
         dist += (x[i] - y[i]) * (x[i] - y[i]);
      return dist;
   }


   /** 
    * Calculates WCSS (Within-Cluster-Sum-of-Squares), a measure of the clustering's error.
    */
   private void calcWCSS() {
      double WCSS = 0;
      int assignedClust;

      for (int i = 0; i < m; i++) {
         assignedClust = assignment[i];
         WCSS += EuclideanDistSq(points[i], centroids[assignedClust]);
      }     

      this.WCSS = WCSS;
   }

   /***********************************************************************
    * Accessors
    ***********************************************************************/
   public int[] getAssignment() {
      return assignment;
   }

   public double[][] getCentroids() {
      return centroids;
   }

   public double getWCSS() {
      return WCSS;
   }



   /***********************************************************************
    * Unit testing
    **********************************************************************/
   
   public static void main(String args[]) throws IOException {
      // the test data is four 750-point Gaussian clusters (3000 points in all)
      // created around the vertices of the unit square
      String data = "TestData.csv";
      int numPoints = 3000;
      int dimensions = 2;
      int k = 4;
      double[][] points = CSVreader.read(data, numPoints, dimensions);

      // run K-means
      final long startTime = System.currentTimeMillis();
      KMeans clustering = new KMeans.Builder(k, points)
                                    .iterations(50)
                                    .pp(true)
                                    .epsilon(.001)
                                    .useEpsilon(true)
                                    .build();
      final long endTime = System.currentTimeMillis();
     
      // print timing information
      final long elapsed = endTime - startTime;
      System.out.println("Clustering took " + (double) elapsed/1000 + " seconds");
      System.out.println();

      // get output
      double[][] centroids = clustering.getCentroids();
      double WCSS 		   = clustering.getWCSS();
      // int[] assignment  = kmean.getAssignment();

      // print output
      for (int i = 0; i < k; i++)
         System.out.println("(" + centroids[i][0] + ", " + centroids[i][1] + ")");
      System.out.println();

      System.out.println("The within-cluster sum-of-squares (WCSS) = " + WCSS);
      System.out.println();
      
      // write output to CSV
      // CSVwriter.write("filePath", centroids);
   }

}