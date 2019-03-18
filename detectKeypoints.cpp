


#include <chrono>
#include <sys/time.h>

// extra headers for writing out ply file
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

//reconstruction headers
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>


#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/registration/icp.h>

#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>

#include <pcl/keypoints/sift_keypoint.h>

#include "detectKeypoints.h"

pcl::PointCloud<pcl::PointNormal>::Ptr detectKeypoints (const pcl::PointCloud<pcl::PointXYZ>& points, float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr scene (new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(points, *scene);

    // Estimate normals for scene
    pcl::console::print_highlight ("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<pcl::PointNormal,pcl::PointNormal> nest;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree_scene(new pcl::search::KdTree<pcl::PointNormal>());
    nest.setRadiusSearch(0.015);//15
    nest.setInputCloud(scene);
    nest.setSearchMethod(tree_scene);
    nest.compute(*scene);

    {
    pcl::ScopeTime t("Keypoints ");
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift_detect;sift_detect.setRadiusSearch(0.015);
    pcl::PointCloud<pcl::PointWithScale> keypoints_temp;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal>);
    sift_detect.setSearchMethod (tree);
    sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast (min_contrast);
    sift_detect.setInputCloud (scene);
    sift_detect.compute (keypoints_temp);
    pcl::PointCloud<pcl::PointNormal>::Ptr keypoints (new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud (keypoints_temp,* keypoints);
    std::cout << "No of SIFT points in the result are " << keypoints->points.size () << std::endl;
    return (keypoints);
    }

}
