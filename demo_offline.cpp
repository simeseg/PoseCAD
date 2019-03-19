
#include <chrono>
#include <sys/time.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

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
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/fpfh_omp.h>

//registration headers
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/io/vtk_lib_io.h>

#ifdef WITH_SERIALIZATION
#include "serialization.h"
#endif

#include "EuclideanCluster.h"
#include "detectKeypoints.h"
#include "SAC_cluster.h"
#include "rsGrabber.h"
#include <time.h>

// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;


// Aligning a rigid scene with clutter and clustering


using namespace pcl;

int main(int argc, char * argv[])
{

      srand (time(NULL));
      
        // Load object
        pcl::PointCloud<pcl::PointXYZ>::Ptr object (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::console::print_highlight ("Loading object point clouds...\n");
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *object) < 0 )
        {
          pcl::console::print_error ("Error loading object/scene file!\n");
          return (1);
        }

        //load CAD
        pcl::PolygonMesh mesh7; //pcl::io::loadOBJFile("3DScan_test3a.obj", mesh7);
        if(pcl::io::loadPolygonFileOBJ(argv[2], mesh7) < 0 ) {
              pcl::console::print_error ("Error loading CAD mesh file \n ");
              
        }

        // Downsample object and convert to PointNT cloud
        pcl::console::print_highlight ("Downsampling...\n");
        const float leaf = 0.01f;
        pcl::VoxelGrid<pcl::PointXYZ> grid_object;
        grid_object.setLeafSize (leaf, leaf, leaf);
        grid_object.setInputCloud (object);
        grid_object.filter (*object);

        //load frame
      boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> cloud; cloud = object;

      //cloud->sensor_orientation_.w() = 0.0;
      //cloud->sensor_orientation_.x() = 1.0;
      //cloud->sensor_orientation_.y() = 0.0;
      //cloud->sensor_orientation_.z() = 0.0;

      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
      viewer->setBackgroundColor (0, 0, 0);
;
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
      //viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb , "sample cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

      pcl::PointCloud<pcl::PointXYZ>::Ptr scene_tmp (new pcl::PointCloud<pcl::PointXYZ>);
      //viewer->addPointCloud<pcl::PointXYZ>(scene_tmp, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (scene_tmp, 255.0, 0.0, 0.0), "scene keypoints");
      viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene keypoints");

      while(!viewer->wasStopped()){

        viewer->spinOnce ();
        // Showing only color since depth is float and needs conversion
        int c = cv::waitKey(1);

        pcl::PassThrough<pcl::PointXYZRGB> filter;
        filter.setInputCloud(cloud);
        filter.setFilterFieldName("z");
        filter.setFilterLimits(0.0,1.5);
        filter.filter(*cloud);
        filter.setFilterFieldName("x");
        filter.setFilterLimits(-0.25,0.25);
        filter.filter(*cloud);
        filter.setFilterFieldName("y");
        filter.setFilterLimits(-0.25,0.25);
        filter.filter(*cloud);
        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);

        //remove outliers
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1);
        sor.filter(*cloud);

        // Downsample
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::console::print_highlight ("Downsampling...\n");
        pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> grid;
        grid.setLeafSize (leaf, leaf, leaf);
        grid.setInputCloud (cloud);
        grid.filter (*filtered_cloud);

        //cluster
        pcl::copyPointCloud(*filtered_cloud,*scene_tmp);
        std::vector<pcl::PointCloud<pcl::PointXYZ>> cluster;

        cluster = EuclideanCluster(scene_tmp);
        std::cout <<cluster.size() <<"  clusters found \n";

        // Parameters for sift computation
        const float min_scale = 0.008f;//0.0025
        const int n_octaves = 8;//8
        const int n_scales_per_octave = 8;//8
        const float min_contrast = 0.00f;//0.003
        viewer->removeAllPointClouds();
            
        for(unsigned it = 0; it<cluster.size();it++ ){
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_copy (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::copyPointCloud(cluster.at(it), *cluster_copy);

            pcl::PointCloud<pcl::PointNormal>::Ptr cluster_keypoints = detectKeypoints(*object, min_scale, n_octaves, n_scales_per_octave, min_contrast);
            std::stringstream ss;
            ss << "keypoints for cluster " << it;

            Eigen::Matrix4f transform;
            transform = SAC_cluster(cluster.at(it), *object);

            pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*object,*object_aligned,transform);

            //Generalized ICP
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
            gicp.setInputSource(object_aligned);
            gicp.setInputTarget(cluster_copy);
            gicp.setMaxCorrespondenceDistance(0.015);//gicp.setTransformationEpsilon(1e-3);
            gicp.setMaximumIterations(500);
            pcl::PointCloud<pcl::PointXYZ>::Ptr Final (new pcl::PointCloud<pcl::PointXYZ>); gicp.align(*Final);
            std::cout << " ICP has converged:" << gicp.hasConverged() << " score: " <<
            gicp.getFitnessScore() ;

            viewer->addPointCloud<pcl::PointXYZ> (cluster_copy, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (cluster_copy, rand() % 255, rand() % 255, rand() % 255), ss.str());
            viewer->addPointCloud<pcl::PointNormal> (cluster_keypoints, ColorHandlerT (cluster_keypoints, rand() % 255, rand() % 255, rand() % 255), ss.str());
            viewer->addPointCloud<pcl::PointNormal> (cluster_keypoints, ColorHandlerT (cluster_keypoints, 255, 0, 0), ss.str());
            viewer->addPointCloud<pcl::PointXYZ> (object, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (object, 0, 255, 0), "object");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, ss.str());
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "object");
        }
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
        cv::resize(color, color, cv::Size(1080, 840));
        cv::imshow("color",color);
        //viewer->addPolygonMesh(mesh7);
      }

      return EXIT_SUCCESS;
}
