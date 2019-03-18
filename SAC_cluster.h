#ifndef SAC_CLUSTER_H_INCLUDED
#define SAC_CLUSTER_H_INCLUDED

//declaration of Eulidean clustering
Eigen::Matrix4f SAC_cluster (const pcl::PointCloud<pcl::PointXYZ>& cluster, const pcl::PointCloud<pcl::PointXYZ>& object_raw);

#endif
