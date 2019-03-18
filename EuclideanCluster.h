#ifndef EUCLIDEANCLUSTER_H_INCLUDED
#define EUCLIDEANCLUSTER_H_INCLUDED

//declaration of Eulidean clustering
std::vector<pcl::PointCloud<pcl::PointXYZ>> EuclideanCluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

#endif
