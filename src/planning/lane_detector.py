
# ald_boundary_estimator.py -- Estimator class for storing ALD algorithm state

import numpy as np
from os import path
import sys

cone_dead_zone = .7

class AldDebug:
    """ This is a placeholder to make it easy to pass debugging information from the algorithm to ROS """
    def __init__(self):
        pass


class AldBoundaryEstimator:


    def __init__(self, inner_head_pos, inner_head_tangent, outer_head_pos, outer_head_tangent):
        """ Initialize the estimator with inner/outer cone positions and track tangent estimate

            The algorithm relies on part of the previous estimate in each update, so we
            need to give an initial guess. the head_pos variables represent a guess for where
            the two sides of the track are (e.g. the center of the car +/- a lateral offset).
            The _head_tangent variables represent the tangent estimate for the inside/outside
            of the track, which can be initialized to the car's current orientation.

            Args:
                inner_head_pos (1x2 np array): estimate for first track boundary position
                inner_head_tangent (1x2 np array): estimate for first track boundary tangent
                outer_head_pos (1x2 np array): estimate for second track boundary position
                outer_head_tangent (1x2 np array): estimate for second track boundary tangent
        """

        self.inner_head_pos = inner_head_pos
        self.inner_head_tangent = inner_head_tangent
        self.outer_head_pos = outer_head_pos
        self.outer_head_tangent = outer_head_tangent


    def update(self, cones: np.ndarray, allow_closure: bool):
        """ Updates estimate of track boundaries.

            Args:
                cones (N x 2 numpy array): x,y positions of all visible cones
                allow_closure (bool): Is the boundary allowed to loop back onto itself?

            Returns:
                inner_ix_list ([int]): indices into cones array for the inner boundary
                outer_ix_list ([int]): indices into cones array for the outer boundary
                debug: struct with inner_weights and outer_weights lists
        """
        return self.update_numpy(cones, allow_closure)


    def calculate_features(self, sample, head, tan):
        """ Calculates a distance and angle cost for adding point p

            Calculates the cost for an edge from head to p, with current tangent tan

            Args:
                sample (Nx2 np array): positions of candidate nodes
                head (1x2 np array): source of candidate edge
                tan (1x2 np array): unit tangent vector of track at head

            Returns:
                (Nx2 np array, Nx1 np array, Nx1 np array): returns the displacement, distance cost and angle cost
        """
        # compute distance weight to each cone
        displacement_feature = sample - head
        distance_feature = np.linalg.norm(displacement_feature, axis=1, keepdims=True) + 10e-5

        # compute angle weight to each cone
        # Note the first check about distances is so that if cone position changes slightly between estimator updates
        # the seed heads that start the edge growing process, which are saved from the previous iteration, don't end up
        # slightly offset with very large angle.
        # This check could be eliminated if the initial edge head automatically snapped to the closest cone
        #cone_dead_zone = 0.01 # TODO: Move parameter to a better location?
        not_in_cone_dead_zone = distance_feature > cone_dead_zone
        angle_feature = not_in_cone_dead_zone * ((np.inner(tan, displacement_feature / distance_feature) - 1)**2).reshape(len(sample), 1)

        return displacement_feature, distance_feature, angle_feature

    def calculate_features_of_features(self, sample, tan):
        """ Calculates the features of every cone in sample with respect to each cone in sample
        
        Args:
            sample (Nx2 np array): position of candidate nodes
            tan (Nx2 np array): tangent vector of every cone in sample with respect to current head
        
        Returns:
            (NxNx1, NXNx1): 2D array of distance weights, 2D array of angle weights
        """
        sample_size = len(sample)
        displacement_feature = sample - np.repeat(sample, sample_size, axis=0).reshape((sample_size, sample_size, 2)) # NxNx2 array of displacements between each cone
        
        distance_feature = np.linalg.norm(displacement_feature, axis=2, keepdims=True) # NxNx1 array
        distance_feature = np.where(distance_feature==0, 10e5, distance_feature) # Give high weight to distance of zero, i.e. the same cone

        not_in_cone_dead_zone = distance_feature > cone_dead_zone
        angle_feature = np.zeros((len(sample), len(sample), 1))
        for i in range(len(sample)):
            feature = ((np.inner(tan[i], displacement_feature[i] / distance_feature[i]) - 1)**2).reshape(len(sample), 1)
            angle_feature[i] = feature
        angle_feature = not_in_cone_dead_zone * angle_feature
        
        return distance_feature, angle_feature

    def update_numpy(self, sample, allow_closure):
        """ Updates estimate of track boundaries

            Args:
                sample (N x 2 numpy array): x,y location of each cone
                allow_closure (bool): Is the boundary allowed to loop back onto itself?

            Returns:
                inner_ix_list ([Int]): indices of cones on inner edge. This is index of cone in sample, NOT cone's detection ID
                outer_ix_list ([Int]): indices of cones on outer edge. This is index of cone in sample, NOT cone's detection ID
                debug: Struct with debugging/tuning information that can be passed to ROS land
                
        """
        # Think of the inner and outer edges each being snakes
        # that eat the lowest-weight available cone. The head of
        # the snake does the eating :)
        ihead = self.inner_head_pos
        itan = self.inner_head_tangent
        ohead = self.outer_head_pos
        otan = self.outer_head_tangent

        # These lists will contain the ordering of cones contained in the inner and outer wall
        ilist = []
        olist = []

        inner_ix_list = []
        outer_ix_list = []

        n_cones = len(sample)
        cone_ix_avail = list(range(n_cones))

        closure_appended = False

        # Memory for weights from inner/outer head to each other cone
        distances_inner = np.zeros(n_cones)
        distances_outer = np.zeros(n_cones)
        angweights_inner = np.zeros(n_cones)
        angweights_outer = np.zeros(n_cones)
        totweights_inner = np.zeros(n_cones)
        totweights_outer = np.zeros(n_cones)

        totweights_inner2 = np.zeros(n_cones)
        totweights_outer2 = np.zeros(n_cones)

        # not currently used
        colorweights_inner = np.zeros(n_cones)
        colorweights_inner = np.zeros(n_cones)

        # Tuning parameters
        ## Important: These may need to be tuned for better outlier rejection
        alpha = 1 # Tunable scale factor for penalizing distance between cones
        beta = 20 # Tunable scale factor for penalizing angle between cones
        #eta = 1 # This would be used to tune the weight for detected color
        max_weight = 13 # Edges above this weight will never be included in graph.
        #Note that max_weight is only relevant if there are outlier cones included in the sample.
        # If all cones are inliers, max_weight should be large enough that nothing is excluded

        # These store the weights for each edge that was chosen. Returned for use in debugging/tuning
        inner_weights = []
        outer_weights = []

        # 1) process edges
        #    a) compute weights to all (unclaimed) cones in sample
        #    b) if the weight is low enough, move the "head" of the edge chain, "claim" cone (cone cannot be connected again in the future)
        #    c) repeat
        #cone_dead_zone = 0.01

        while(1):
            
            if allow_closure and not closure_appended and min(len(ilist), len(olist)) > 5:
                cone_ix_avail.append(inner_root)
                cone_ix_avail.append(outer_root)
                closure_appended = True

            # Determine weights of ith sample cone with respect to current head
            inner_displacement, w_dist_inner, w_ang_inner = self.calculate_features(sample[cone_ix_avail], ihead, itan)
            outer_displacement, w_dist_outer, w_ang_outer = self.calculate_features(sample[cone_ix_avail], ohead, otan)

            if len(cone_ix_avail) > 1:
                # Determine heading tangent vector of every cone in sample with respect to current head, Nx2 arrays
                itan2 = np.where(w_dist_inner > .1, inner_displacement / w_dist_inner, itan)
                otan2 = np.where(w_dist_outer > .1, outer_displacement / w_dist_outer, otan)

                # NxNx1, NxNx1 arrays of weights 
                w_dist_inner2, w_ang_inner2 = self.calculate_features_of_features(sample[cone_ix_avail], itan2)
                w_dist_outer2, w_ang_outer2 = self.calculate_features_of_features(sample[cone_ix_avail], otan2)
                
                # 2-step cone lookahead
                totweights_inner2 = alpha * w_dist_inner2 + beta * w_ang_inner2
                totweights_outer2 = alpha * w_dist_outer2 + beta * w_ang_outer2

                inner_min2 = np.min(totweights_inner2, axis=1)
                outer_min2 = np.min(totweights_outer2, axis=1)
                
                totweights_inner = ((alpha * w_dist_inner + beta * w_ang_inner) + inner_min2) / 2
                totweights_outer = ((alpha * w_dist_outer + beta * w_ang_outer) + outer_min2) / 2
                
            else:
                totweights_inner = alpha * w_dist_inner + beta * w_ang_inner
                totweights_outer = alpha * w_dist_outer + beta * w_ang_outer
                
            # smallest-weight edge for inner/outer head to eat respectively
            inner_min = np.min(totweights_inner)
            outer_min = np.min(totweights_outer)

            # there are no more inlier cones
            if min(inner_min, outer_min) > max_weight:
                break

            if inner_min < outer_min:
                # inner head should eat a cone
                inner_next_head_ix = cone_ix_avail[np.argmin(totweights_inner)]
                cone_ix_avail.remove(inner_next_head_ix)
                if len(ilist) == 0:
                    inner_root = inner_next_head_ix
                ilist.append(sample[inner_next_head_ix])
                inner_ix_list.append(inner_next_head_ix)
                if np.linalg.norm(sample[inner_next_head_ix] - ihead) > cone_dead_zone:
                    itan = (sample[inner_next_head_ix] - ihead) / np.linalg.norm(sample[inner_next_head_ix] - ihead)
                ihead = sample[inner_next_head_ix]

                inner_weights.append(inner_min)
            else:
                #outer head should eat a cone
                outer_next_head_ix = cone_ix_avail[np.argmin(totweights_outer)]
                cone_ix_avail.remove(outer_next_head_ix)
                if len(olist) == 0:
                    outer_root = outer_next_head_ix
                olist.append(sample[outer_next_head_ix])
                outer_ix_list.append(outer_next_head_ix)
                if np.linalg.norm(sample[outer_next_head_ix] - ohead) > cone_dead_zone:
                    otan = (sample[outer_next_head_ix] - ohead) / np.linalg.norm(sample[outer_next_head_ix] - ohead)
                ohead = sample[outer_next_head_ix]
                outer_weights.append(outer_min)

            if len(inner_ix_list) > 1 and inner_ix_list[0] == inner_ix_list[-1] and len(outer_ix_list) > 1 and outer_ix_list[0] == outer_ix_list[-1]:
                break

            if len(cone_ix_avail) < 1:
                break
        # 2) address any conflicts
        # TODO: Here we should do some checks to make sure the result is reasonable, e.g. ensure the
        # distance between inner and outer edge is larger enough, and that the local tangent of inner and outer edges are similar

        # 3) update head positions if necessary. This updates the head "seeds" that will be used in the next iteration to start
        # building the edge chains
        if ilist:
            if np.linalg.norm(self.inner_head_pos - ilist[0]) > .5:
                if len(ilist) > 1:
                    self.inner_head_tangent = (ilist[1] - ilist[0]) / np.linalg.norm(ilist[1] - ilist[0])
                self.inner_head_pos = ilist[0]

        if olist:
            if np.linalg.norm(self.outer_head_pos - olist[0]) > .5:
                if len(olist) > 1:
                    self.outer_head_tangent = (olist[1] - olist[0]) / np.linalg.norm(olist[1] - olist[0])
                self.outer_head_pos = olist[0]
        # we don't want to always update the head to the furthest observed cone.
        # it should be the cone ~closest to the car

        debug = AldDebug()
        debug.inner_weights = inner_weights
        debug.outer_weights = outer_weights

        return inner_ix_list, outer_ix_list, debug
