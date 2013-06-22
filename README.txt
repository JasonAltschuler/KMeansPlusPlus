/*********************************************************************************
* Description of KMeans.java
*********************************************************************************/

   PURPOSE: Clusters n-dimensional points via an improved form of K-Means algorithm.

   OVERVIEW: Randomly chooses without replacement k (# of clusters) data points to be
initial centroids, either by basic random sampling or by KMeans++ (described below).
Then runs clustering multiple times and chooses the best run (based on WCSS, a measure 
of goodness-of-fit). Clustering consists of repeating 2 alternating steps: the
assignment step and the update step.  In the assignment step, all data points are 
assigned to the nearest centroid. In the update step, centroids are recalculated based on
what points are assigned to them.  These two steps are repeated until the marginal 
improvement in WCSS is either 0 or less than 'epsilon' (provided by user, or default = .001).

   DESCRIPTION OF KMEANS++: An improved method for randomly sampling the initial centroids.
Iteratively chooses the next centroid based on a weighted distribution, where
the probability a data point is chosen is directly proportional to the square 
of the distance between that point and the closest centroid. 

   PARAMETERS / DATA STRUCTURES: for more description, see the documentation in the code itself.
Default and suggested values are provided directly below.

   REQUIRED PARAMETERS:
int k                -- number of clusters. Recommended: fewer is better, as long as the fit is decent.
                     See: http://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set
double[][] points    -- n-dimensional data points. For PhenoRipper, this is pixels by channels. 
                     Entries stores pixel intensity.
    
   OPTIONAL PARAMETERS: default and suggested values
int iterations       -- Default: 50. Recommended [50, 1000]
boolean pp           -- Default: true. Recommended: true. Plus-plus is more precise and 
                     potentially faster than basic random sampling, especially if epsilon is small.
double epsilon       -- Default: .001. Recommended: [.0001 to .01]. Smaller -> more precision.
                     Larger -> faster. Remember to set useEpsilon to true if use this parameter.
boolean useEpsilon   -- Default: true. Recommended unless extremely exact results needed.
                     value of false could potentially make KMeans much slower.
  
	MISC DATA STRUCTURES: these are calculated from dimension of points[][]
int m                -- Number of data points. Typically denoted by 'm' in machine-learning.
                     In PhenoRipper, this is the number of pixels in the image.
int n                -- Number of dimensions. Typically denoted by 'n' in machine-learning.
                     In PhenoRipper, this is the number of channels in the images.

   OUTPUT:
double[][] centroids -- the centroids of the clusters.
int[] assignment     -- denotes which points belong to which clusters.
double WCSS          -- Within-Cluster-Sum-Of-Squares. Measure of goodness-of-fit / error.

   SUGGESTED READINGS:  http://en.wikipedia.org/wiki/K-means_clustering
                        http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html

   EXAMPLE RUN: 
double[][] points = User.provideData();         // you need to provide
int k = User.provideNumberOfCluster();          // you need to provide

KMeans example = new KMeans.builder(k, points)  // required
                     .iterations(50)            // optional
                     .pp(true)                  // optional
                     .epsilon(.001)             // optional
                     .useEpsilon(true)          // optional
                     .build();                  // required

double[][] centroids = example.getCentroids();  // centroid of clusters
int[] assignment     = example.getAssignment(); // which cluster each point belongs to
double WCSS          = example.getWCSS();       // goodness-of-fit

User.do_something_awesome_with_the_data();      // enjoy!