package com.dsp.haptrix;

import java.util.ArrayList;

/**
 * Interface for the implementation of distance metrics.
 * 
 * @author <a href="mailto:cf@christopherfrantz.org">Christopher Frantz</a>
 * @version 0.1
 *
 * @param <V> Value type to which distance metric is applied.
 */
public interface DistanceMetric<V> {

    public double calculateDistance(ArrayList V1, ArrayList V2) throws DBSCANClusteringException;

}



