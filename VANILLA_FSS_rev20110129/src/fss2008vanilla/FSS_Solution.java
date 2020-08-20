package fss2008vanilla;

import cec2010suite.Function;


/*
 * Copyright (c) 2011 Murilo Rebelo Pontes
 * murilo.pontes@gmail.com
 * 
 * GNU LESSER GENERAL PUBLIC LICENSE (Version 2.1, February 1999)
 */



public class FSS_Solution {
	//
	public double[] variables = null;
	public double fitness = Double.NaN;
	//
	public FSS_Solution(int dims) {
		variables = new double[dims];
	}
	public static void copy(FSS_Solution source,FSS_Solution destination){
		//
		destination.fitness = source.fitness;
		//
		for(int i=0;i<source.variables.length;i++){
			destination.variables[i]=source.variables[i];
		}
	}
	public static void randomize_variables(FSS_Solution sol,FSS_PRNG r,Function f){
		for(int i=0;i<sol.variables.length;i++){
			sol.variables[i]=r.between(f.getMin(), f.getMax());
		}
	}
}
