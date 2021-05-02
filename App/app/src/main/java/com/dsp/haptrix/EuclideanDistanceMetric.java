package com.dsp.haptrix;

import java.util.ArrayList;

public class EuclideanDistanceMetric implements DistanceMetric<Number> {

    @Override
    public double calculateDistance(ArrayList val1, ArrayList val2) {
        Float val1_x = (Float) val1.get(0);
        Float val1_y = (Float) val1.get(1);
        Float val2_x = (Float) val2.get(0);
        Float val2_y = (Float) val2.get(1);
        return Math.sqrt(Math.pow(val1_x - val2_x, 2) + Math.pow(val1_y - val2_y, 2));
    }

}
