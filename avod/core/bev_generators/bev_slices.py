import numpy as np

from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

from avod.core.bev_generators import bev_generator


class BevSlices(bev_generator.BevGenerator):

    NORM_VALUES = {
        'lidar': np.log(16),
        'distance': 1
    }

    def __init__(self, config, kitti_utils):
        """BEV maps created using slices of the point cloud.

        Args:
            config: bev_generator protobuf config
            kitti_utils: KittiUtils object
        """

        # Parse config
        self.height_lo = config.height_lo
        self.height_hi = config.height_hi
        self.num_slices = config.num_slices
        self.slice_maps = config.slice_maps
        self.cloud_maps = config.cloud_maps

        # If nothing specified uses the normal avod config
        if (len(self.slice_maps) == 0 and len(self.cloud_maps) == 0):
            self.slice_maps = ['max']
            self.cloud_maps = ['density']

        print("slice maps: ", self.slice_maps)
        print("cloud maps: ", self.cloud_maps)

        self.kitti_utils = kitti_utils

        # Pre-calculated values
        self.height_per_division = (self.height_hi - self.height_lo) / self.num_slices


    def generate_bev_map(self, voxel_grid_2d, map_config, map_container, height_lo, height_hi, area_extents, source):
        slice_height = height_hi - height_lo
        num_divisions = voxel_grid_2d.num_divisions

        # Remove y values (all 0)
        voxel_indices = voxel_grid_2d.voxel_indices[:, [0, 2]]

        if "max" in map_config:
            height_map = self.get_height_map(voxel_grid_2d.heights, height_lo, slice_height, num_divisions, voxel_indices)
            map_container.append(height_map)

        if "min" in map_config:
            min_height_map = self.get_height_map(voxel_grid_2d.min_heights, height_lo, slice_height, num_divisions, voxel_indices)
            map_container.append(min_height_map)

        if "variance" in map_config:
            #Create empty BEV image
            variance_map = np.zeros((voxel_grid_2d.num_divisions[0],
                                voxel_grid_2d.num_divisions[2]))

            # Only update pixels where voxels have values
            variance_map[voxel_indices[:, 0], voxel_indices[:, 1]] = np.asarray(voxel_grid_2d.variance) # np.multiply(voxel_grid_2d.variance, voxel_grid_2d.num_pts_in_voxel))
            variance_map /= np.max(variance_map)
            variance_map = np.flip(variance_map.transpose(), axis=0)

            map_container.append(variance_map)

        if "density" in map_config or "dnd" in map_config:
            if "dnd" in map_config: # Can't use both density and dnd in a single model like this
                # Normalize distances by the maximum extent
                self.NORM_VALUES["distance"] = voxel_grid_2d.distances / np.max(area_extents)

            density_map = self._create_density_map(
                num_divisions=voxel_grid_2d.num_divisions,
                voxel_indices_2d=voxel_indices,
                num_pts_per_voxel=voxel_grid_2d.num_pts_in_voxel,
                norm_value=self.NORM_VALUES[source],
                distance=self.NORM_VALUES["distance"])

            map_container.append(density_map)

        if "cluster" in map_config:
            # Remove y values (all 0)
            cluster_voxel_indices = voxel_grid_2d.cluster_voxel_indices[:, [0, 2]]
            
            # Add highest point of all clusters
            cluster_height_map = self.get_height_map(voxel_grid_2d.cluster_heights, height_lo, slice_height, num_divisions, cluster_voxel_indices)
            map_container.append(cluster_height_map)

            # Add lowest point of all clusters
            cluster_min_height_map = self.get_height_map(voxel_grid_2d.cluster_min_heights, height_lo, slice_height, num_divisions, cluster_voxel_indices)
            map_container.append(cluster_min_height_map)

            cluster_density_map = self._create_density_map(
                num_divisions=voxel_grid_2d.num_divisions,
                voxel_indices_2d=cluster_voxel_indices,
                num_pts_per_voxel=voxel_grid_2d.num_pts_in_cluster,
                norm_value=self.NORM_VALUES[source],
                distance=self.NORM_VALUES["distance"])

            map_container.append(cluster_density_map)
            

    def generate_bev(self,
                     source,
                     point_cloud,
                     ground_plane,
                     area_extents,
                     voxel_size):
        """Generates the BEV maps dictionary. Generates specified feature maps for each slice or the whole point cloud.

        Args:
            source: point cloud source
            point_cloud: point cloud (3, N)
            ground_plane: ground plane coefficients
            area_extents: 3D area extents
                [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
            voxel_size: voxel size in m

        Returns:
            BEV maps dictionary
                slice_maps: list of feature maps per slice
                cloud_maps: list of feature maps for the whole point cloud
        """

        all_points = np.transpose(point_cloud)

        slice_maps = []
        cloud_maps = []

        if len(self.slice_maps) > 0:
            for slice_idx in range(self.num_slices):

                height_lo = self.height_lo + slice_idx * self.height_per_division
                height_hi = height_lo + self.height_per_division

                slice_filter = self.kitti_utils.create_slice_filter(
                    point_cloud,
                    area_extents,
                    ground_plane,
                    height_lo,
                    height_hi)

                # Apply slice filter
                slice_points = all_points[slice_filter]

                if len(slice_points) >= 0:  # Should probably apply the fix for empty BEV slices
                    # Create Voxel Grid 2D
                    voxel_grid_2d = VoxelGrid2D()
                    voxel_grid_2d.voxelize_2d(
                        slice_points, voxel_size,
                        extents=area_extents,
                        ground_plane=ground_plane,
                        create_leaf_layout=False,
                        maps=self.slice_maps)

                    self.generate_bev_map(voxel_grid_2d, self.slice_maps, slice_maps, height_lo, height_hi, area_extents, source)

        if len(self.cloud_maps) > 0:
            cloud_filter = self.kitti_utils.create_slice_filter(
                point_cloud,
                area_extents,
                ground_plane,
                self.height_lo,
                self.height_hi)

            cloud_points = all_points[cloud_filter]

            if len(cloud_points >= 0):
                # Create Voxel Grid 2D
                voxel_grid_2d = VoxelGrid2D()
                voxel_grid_2d.voxelize_2d(
                    cloud_points,
                    voxel_size,
                    extents=area_extents,
                    ground_plane=ground_plane,
                    create_leaf_layout=False,
		            maps=self.cloud_maps)

                self.generate_bev_map(voxel_grid_2d, self.cloud_maps, cloud_maps, self.height_lo, self.height_hi, area_extents, source)

        bev_maps = dict()
        bev_maps['slice_maps'] = slice_maps
        bev_maps['cloud_maps'] = cloud_maps

        return bev_maps

    def get_height_map(self, heights, height_lo, slice_height, num_divisions, voxel_indices):
        # Create empty BEV image
        height_map = np.zeros((num_divisions[0], num_divisions[2]))

        # Only update pixels where voxels have height values,
        # and normalize by height of slices
        norm_heights = heights - height_lo
        height_map[voxel_indices[:, 0], voxel_indices[:, 1]] = np.asarray(norm_heights) / slice_height

        # Rotates slice map 90 degrees
        # (transpose and flip) is faster than np.rot90
        height_map = np.flip(height_map.transpose(), axis=0)

        return height_map
