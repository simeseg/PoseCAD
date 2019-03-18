#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <pcl/point_types.h>
#include <Eigen/Dense>
//reconstruction headers
#include <pcl/filters/voxel_grid.h>
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

#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/keyboard_event.h>

//ransac
#include <pcl/console/parse.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include </usr/local/include/pcl-1.9/pcl/gpu/octree/octree.hpp>

#include <chrono>
#include <sys/time.h>
#include <iostream>

#include "EuclideanCluster.h"
#include "detectKeypoints.h"
#include "SAC_cluster.h"
#include <time.h>

using namespace pcl;
using namespace std;
using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointcloud);
void ViewFrame(pcl_ptr cloud, pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr initializer_cloud);

pcl_ptr points_to_pcl(const rs2::points& points)
 {
     pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

     auto sp = points.get_profile().as<rs2::video_stream_profile>();
     cloud->width = sp.width();
     cloud->height = sp.height();
     cloud->is_dense = false;
     cloud->points.resize(points.size());
     auto ptr = points.get_vertices();
     for (auto& p : cloud->points)
     {
         p.x =  ptr->x;
         p.y =  ptr->y;
         p.z =  ptr->z;
         ptr++;
     }

     return cloud;
 }


 void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void){
     boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> ptcloud = *static_cast<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> *>(viewer_void);
     std::string pressed = event.getKeySym();

       if(event.keyDown ())
       {
         if(pressed == "s")
         {

           std::cout<<"saving pcd file \n";

           std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();
           std::vector<int> indices;
           pcl::removeNaNFromPointCloud(*ptcloud,*ptcloud,indices);
           pcl::io::savePCDFileASCII("scene_model_cloud.pcd", *ptcloud);

           std::cout << "saved sample_cloud.pcd \n";
         }
       }

 }
int main ()
{

     srand (time(NULL));

      // Load object
      pcl::PointCloud<pcl::PointXYZ>::Ptr object (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::console::print_highlight ("Loading object point clouds...\n");
      if (pcl::io::loadPCDFile<pcl::PointXYZ> ("hexagon_model.pcd", *object) < 0 )
      {
        pcl::console::print_error ("Error loading object/scene file!\n");
        return (1);
      }

      //load CAD
      pcl::PolygonMesh mesh7;
      pcl::io::loadPolygonFileOBJ("Hexagon.obj", mesh7);
      //pcl::io::loadOBJFile("3DScan_test3a.obj", mesh7);

      // Downsample object and convert to PointNT cloud
      pcl::console::print_highlight ("Downsampling...\n");
      const float leaf = 0.01f;
      pcl::ApproximateVoxelGrid<pcl::PointXYZ> grid_object;
      grid_object.setLeafSize (leaf, leaf, leaf);
      grid_object.setInputCloud (object);
      grid_object.filter (*object);

     pcl::PointCloud<pcl::PointXYZ>::Ptr initializer_cloud(new pcl::PointCloud<pcl::PointXYZ>);
     //initializer_cloud->sensor_orientation_.w() = 0.0;
     //initializer_cloud->sensor_orientation_.x() = 1.0;
     //initializer_cloud->sensor_orientation_.y() = 0.0;
     //initializer_cloud->sensor_orientation_.z() = 0.0;

     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
     viewer->setBackgroundColor (0, 0, 0);
     viewer->addPointCloud(initializer_cloud,"cloud");
     viewer->registerKeyboardCallback(KeyboardEventOccurred,(void*)&initializer_cloud);
     //viewer->addPointCloud<pcl::PointXYZ>(scene_tmp, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (scene_tmp, 255.0, 0.0, 0.0), "scene keypoints");
     //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene keypoints");

       //load frame

     // Declare pointcloud object, for calculating pointclouds and texture mappings
     rs2::pointcloud pc;
     // We want the points object to be persistent so we can display the last cloud when a frame drops
     rs2::points points;

     // Declare RealSense pipeline, encapsulating the actual device and sensors
     rs2::pipeline pipe;
     // Start streaming with default recommended configuration
     pipe.start();


     while(!viewer->wasStopped()){
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();
        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);
        pcl_ptr cloud = points_to_pcl(points);
        pcl::console::print_error ("started loading object/scene file!\n");
        //visualize frame
        //ViewFrame(cloud, viewer, initializer_cloud);

        pcl::PassThrough<pcl::PointXYZ> filter;
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


        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1);
        sor.filter(*cloud);

        // Downsample
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::console::print_highlight ("Downsampling...\n");
        pcl::ApproximateVoxelGrid<pcl::PointXYZ> grid;
        grid.setLeafSize (leaf, leaf, leaf);
        grid.setInputCloud (cloud);
        grid.filter (*filtered_cloud);

        std::vector<pcl::PointCloud<pcl::PointXYZ>> cluster;

        cluster = EuclideanCluster(filtered_cloud);
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

            //pcl::PointCloud<pcl::PointNormal>::Ptr cluster_keypoints = detectKeypoints(*object, min_scale, n_octaves, n_scales_per_octave, min_contrast);
            std::stringstream ss;
            ss << "keypoints for cluster " << it;
/*
            Eigen::Matrix4f transform;
            transform = SAC_cluster(cluster.at(it), *object);

            pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*object,*object_aligned,transform);

            //Generalized ICP
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
            gicp.setInputSource(object_aligned);
            gicp.setInputTarget(cluster_copy);
            //gicp.setMaxCorrespondenceDistance(0.015);//gicp.setTransformationEpsilon(1e-3);
            gicp.setMaximumIterations(500);
            pcl::PointCloud<pcl::PointXYZ>::Ptr Final (new pcl::PointCloud<pcl::PointXYZ>); gicp.align(*Final);

            std::cout << " ICP has converged:" << gicp.hasConverged() << " score: " <<
            gicp.getFitnessScore() ;
            //std::cout << gicp.getFinalTransformation() << std::endl;
*/

            viewer->addPointCloud<pcl::PointXYZ> (cluster_copy, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (cluster_copy, rand() % 255, rand() % 255, rand() % 255), ss.str());
            //viewer->addPointCloud<pcl::PointNormal> (cluster_keypoints, ColorHandlerT (cluster_keypoints, rand() % 255, rand() % 255, rand() % 255), ss.str());
            //viewer->addPointCloud<pcl::PointNormal> (cluster_keypoints, ColorHandlerT (cluster_keypoints, 255, 0, 0), ss.str());
            //viewer->addPointCloud<pcl::PointXYZ> (object, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> (object, 0, 255, 0), "object");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, ss.str());
            //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "object");
        }
        viewer->updatePointCloud<pcl::PointXYZ> (cloud, "cloud");
        //viewer->addPolygonMesh(mesh7);
     }

}


 void ViewFrame(pcl_ptr cloud, pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr initializer_cloud){

     //viewer->addPointCloud<pcl::PointXYZ>(cloud);
     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1);

     pcl::PassThrough<pcl::PointXYZ> filter;
     filter.setInputCloud(cloud);
     filter.setFilterFieldName("z");
     filter.setFilterLimits(0.05,4);
     filter.filter(*cloud);
     filter.setFilterFieldName("x");
     filter.setFilterLimits(-0.5,0.5);
     filter.filter(*cloud);
     filter.setFilterFieldName("y");
     filter.setFilterLimits(-0.5,0.5);
     filter.filter(*cloud);

     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color (cloud, 255, 0, 0);
     viewer->updatePointCloud<pcl::PointXYZ> (cloud, color);
     viewer->spinOnce (100);
     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
     *initializer_cloud = *cloud;
 }
