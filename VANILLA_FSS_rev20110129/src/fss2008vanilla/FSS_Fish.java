package fss2008vanilla;

import cec2010suite.Function;

/*
 * Copyright (c) 2011 Murilo Rebelo Pontes
 * murilo.pontes@gmail.com
 * 
 * GNU LESSER GENERAL PUBLIC LICENSE (Version 2.1, February 1999)
 */


public class FSS_Fish {
	
	public FSS_Solution neighbor = null;
	public FSS_Solution current = null;
	public FSS_Solution best = null;
	public double weight_now = Double.NaN;
	public double weight_past = Double.NaN;
	public double[] delta_x = null;
	public double delta_f = Double.NaN;
	public double fitness_gain_normalized = Double.NaN;
	
	private Function f = null;
	private FSS_PRNG r = null;
	public boolean individual_move_success=false;
	public boolean instinctive_move_success = false;
	public boolean volitive_move_success = false;
	
	public FSS_Fish(Function f,FSS_PRNG r) {
		this.f = f;
		this.r = r;
		init();
	}
	
	private void init(){
		//
		current=new FSS_Solution(f.getDimension());
		best=new FSS_Solution(f.getDimension());
		neighbor=new FSS_Solution(f.getDimension());
		
		//set all solution equals
		FSS_Solution.randomize_variables(current, r, f);
		current.fitness=f.compute(current.variables);
		
		FSS_Solution.copy(current, best);
		FSS_Solution.copy(current, neighbor);
		
		//
		delta_x = new double[f.getDimension()];
		
		//init fish weight
		weight_now = r.between(FSS_Parameters.fish_weight_min,FSS_Parameters.fish_weight_max);
		weight_past = weight_now;
	}
	
}
