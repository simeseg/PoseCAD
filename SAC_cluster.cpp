#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "detectKeypoints.h"
#include "SAC_cluster.h"

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimation<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

// Aligning a rigid object to a scene with clutter and occlusions

Eigen::Matrix4f SAC_cluster (const pcl::PointCloud<pcl::PointXYZ>& cluster_raw, const pcl::PointCloud<pcl::PointXYZ>& object_raw){

  const float leaf = 0.01f;

  // Parameters for sift computation
  const float min_scale = 0.008f;//001
  const int n_octaves = 8;
  const int n_scales_per_octave = 8;
  const float min_contrast = 0.00f;//003
  pcl::PointCloud<pcl::PointNormal>::Ptr keypoints_scene = detectKeypoints(cluster_raw,min_scale,n_octaves,n_scales_per_octave,min_contrast);
  pcl::PointCloud<pcl::PointNormal>::Ptr keypoints_object = detectKeypoints(object_raw,min_scale,n_octaves,n_scales_per_octave,min_contrast);

  // Point clouds with normals
  PointCloudT::Ptr object (new PointCloudT);
  pcl::copyPointCloud(object_raw, *object);
  PointCloudT::Ptr scene (new PointCloudT);
  pcl::copyPointCloud(cluster_raw, *scene);

  // Estimate normals for scene and object
  pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<PointNT,pcl::PointNormal> nest;
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_object_normal(new pcl::search::KdTree<pcl::PointNormal>());
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_scene_normal(new pcl::search::KdTree<pcl::PointNormal>());
  nest.setRadiusSearch(1.5*leaf);nest.setNumberOfThreads(8);
  {
      pcl::ScopeTime t("scene normals");
  nest.setInputCloud (scene);
  nest.setSearchMethod(tree_scene_normal);
  nest.compute (*scene);
  }
  nest.setInputCloud(object);
  nest.setSearchMethod(tree_object_normal);
  nest.compute(*object);


  // Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureCloudT::Ptr object_features (new FeatureCloudT);
  FeatureCloudT::Ptr scene_features (new FeatureCloudT);
  FeatureEstimationT fest;
  fest.setRadiusSearch (0.02);
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_scene_pfh(new pcl::search::KdTree<pcl::PointNormal>());
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_object_pfh(new pcl::search::KdTree<pcl::PointNormal>());
  fest.setInputCloud (keypoints_object);fest.setSearchMethod(tree_object_pfh);
  fest.setInputNormals (keypoints_object);//fest.setSearchSurface(object);
  fest.compute (*object_features);

  fest.setInputCloud (keypoints_scene);fest.setSearchMethod(tree_scene_pfh);
  fest.setInputNormals (keypoints_scene);//fest.setSearchSurface(scene);
  {
      pcl::ScopeTime t("Scene Features");
      fest.compute (*scene_features);
  }

  // Perform alignment
  PointCloudT::Ptr object_keypoints_aligned (new PointCloudT);
  pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusInitialAlignment<PointNT,PointNT,FeatureT> sac;
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_scene_sac(new pcl::search::KdTree<pcl::PointNormal>());
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_object_sac(new pcl::search::KdTree<pcl::PointNormal>());
  sac.setInputCloud(keypoints_object);sac.setSearchMethodSource(tree_object_sac);
  sac.setSourceFeatures (object_features);
  sac.setInputTarget (keypoints_scene);sac.setSearchMethodTarget(tree_scene_sac);
  sac.setTargetFeatures (scene_features);
  sac.setMaximumIterations (100); // Number of RANSAC iterations
  sac.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
  sac.setCorrespondenceRandomness (3); // Number of nearest features to use
  sac.setMaxCorrespondenceDistance (1.2f * leaf); // Inlier threshold
  sac.getTransformationEpsilon();
  {
      pcl::ScopeTime t("Alignment");
      sac.align (*object_keypoints_aligned);
    }

  Eigen::Matrix4f transformation = sac.getFinalTransformation ();


  if (sac.hasConverged ())
  {
    // Print results
    printf ("\n");
    std::cout<<sac.getFitnessScore()<<" fitness epsilon \n";
    //pcl::transformPointCloud(*object,*object,transformation);
  }
  else
  {
    pcl::console::print_error ("Alignment failed!\n");
    //return (1);
  }
  return transformation;
}
