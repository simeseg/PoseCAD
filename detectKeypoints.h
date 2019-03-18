//
#ifndef DETECTKEYPOINTS_H_INCLUDED
#define DETECTKEYPOINTS_H_INCLUDED

//declaration of Eulidean clustering
pcl::PointCloud<pcl::PointNormal>::Ptr detectKeypoints (const pcl::PointCloud<pcl::PointXYZ>& points, float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast);

#endif
