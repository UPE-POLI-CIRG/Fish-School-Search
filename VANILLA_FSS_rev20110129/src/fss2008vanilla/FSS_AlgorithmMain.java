package fss2008vanilla;
import java.util.Arrays;
import java.util.LinkedList;

import cec2010suite.F1;
import cec2010suite.F10;
import cec2010suite.F11;
import cec2010suite.F12;
import cec2010suite.F13;
import cec2010suite.F14;
import cec2010suite.F15;
import cec2010suite.F16;
import cec2010suite.F17;
import cec2010suite.F18;
import cec2010suite.F19;
import cec2010suite.F2;
import cec2010suite.F20;
import cec2010suite.F3;
import cec2010suite.F4;
import cec2010suite.F5;
import cec2010suite.F6;
import cec2010suite.F7;
import cec2010suite.F8;
import cec2010suite.F9;
import cec2010suite.Function;

/*
 * Copyright (c) 2011 Murilo Rebelo Pontes
 * murilo.pontes@gmail.com
 * 
 * GNU LESSER GENERAL PUBLIC LICENSE (Version 2.1, February 1999)
 */

public class FSS_AlgorithmMain {



	private static int individual_operator(FSS_Fish[] school,Function f,FSS_PRNG r,double step_size){

		int count_success = 0;

		for(FSS_Fish fish: school){
			//use current as template for neighbor
			FSS_Solution.copy(fish.current, fish.neighbor);

			//calculate displacement 
			for(int i=0;i<f.getDimension();i++){
				fish.delta_x[i]=r.between(-1, 1)*step_size;
			}

			//generate new solution in neighbor
			for(int i=0;i<f.getDimension();i++){
				fish.neighbor.variables[i] += fish.delta_x[i];
				//take care about bounds of search space 
				if(fish.neighbor.variables[i]<f.getMin()) fish.neighbor.variables[i]=f.getMin();
				if(fish.neighbor.variables[i]>f.getMax()) fish.neighbor.variables[i]=f.getMax();
			}

			//evaluate new current solution
			fish.neighbor.fitness = f.compute(fish.neighbor.variables);

			//calculate fitness difference
			fish.delta_f = fish.neighbor.fitness-fish.current.fitness;

			//update current if neighbor is best
			fish.individual_move_success = false;
			if(fish.neighbor.fitness<fish.current.fitness){
				FSS_Solution.copy(fish.neighbor, fish.current);
				fish.individual_move_success = true;
				count_success++;
			}

			//if need replace best solution
			if(fish.current.fitness<fish.best.fitness){
				FSS_Solution.copy(fish.current, fish.best);
			}
		}
		return count_success;
	}

	private static void feeding_operator(FSS_Fish[] school){

		//sort school by fitness gain
		Arrays.sort(school,new FSS_ComparatorByFitnessGain());

		//max absolute value of fitness gain
		double abs_delta_f_max=Math.abs(school[0].delta_f);
		double abs_delta_f_max2=Math.abs(school[school.length-1].delta_f);
		if(abs_delta_f_max2>abs_delta_f_max) abs_delta_f_max=abs_delta_f_max2;

		//take care about zero division
		if(abs_delta_f_max!=0){
			//calculate normalized gain
			for(FSS_Fish fish: school){
				fish.fitness_gain_normalized = fish.delta_f/abs_delta_f_max;
			}

			//feed all fishes
			for(FSS_Fish fish: school) {
				//
				fish.weight_past = fish.weight_now;
				fish.weight_now += fish.fitness_gain_normalized;
				//take care about min and max weight
				if(fish.weight_now<FSS_Parameters.fish_weight_min) fish.weight_now=FSS_Parameters.fish_weight_min; 
				if(fish.weight_now>FSS_Parameters.fish_weight_max) fish.weight_now=FSS_Parameters.fish_weight_max; 
			}
		} 
		else {
			//warning user
			System.err.println("bypass feeding (zero division)");
		}

	}

	private static double[] calculate_instinctive_direction(FSS_Fish[] school,Function f) {
		//
		double[] school_instinctive=new double[f.getDimension()];
		double[] sum_prod=new double[f.getDimension()];
		double sum_fitness_gain = 0;

		//clear
		for(int i=0;i<sum_prod.length;i++){
			sum_prod[i]=0;
			school_instinctive[i]=0;
		}

		//for each fish contribute with your direction scaled by your fitness gain
		for(FSS_Fish fish: school){
			//only good fishes
			if(fish.individual_move_success){
				//sum product of solution by fitness gain 
				for(int i=0;i<fish.delta_x.length;i++){
					sum_prod[i]+= fish.delta_x[i]*fish.fitness_gain_normalized;
				}
				//sum fitness gains
				sum_fitness_gain+=fish.fitness_gain_normalized;
			}
		}

		//calculate global direction of good fishes
		for(int i=0;i<sum_prod.length;i++){
			//take care about zero division
			if(sum_fitness_gain!=0){
				school_instinctive[i]=sum_prod[i]/sum_fitness_gain;
			} 
			else {
				school_instinctive[i]=0;
			}
		}
		return school_instinctive;
	}
	
	private static double[] colletive_instinctive_operator(FSS_Fish[] school,Function f){
		
		//
		double[] school_instinctive = calculate_instinctive_direction(school, f);

		//
		for(FSS_Fish fish: school) {
			//use current as template to neighbor
			FSS_Solution.copy(fish.current, fish.neighbor);

			//update neighbor with instinctive direction
			for(int i=0;i<fish.neighbor.variables.length;i++){
				fish.neighbor.variables[i] += school_instinctive[i];
			}
		}

		return school_instinctive;
	}



	private static double ecliadian_distance(double[] a,double[] b){
		double sum=0;
		for(int i=0;i<a.length;i++){
			sum+=Math.pow(a[i]-b[i],2.0);
		}
		return Math.sqrt(sum);
	}

	private static int colletive_volitive_operator(FSS_Fish[] school,Function f,double step_size,FSS_PRNG r,double[] school_instinctive){

		////////////////////////////////////////////////////////////////////////////////

		double[] school_barycentre=new double[f.getDimension()];
		double[] sum_prod=new double[f.getDimension()];
		double sum_weight_now = 0;
		double sum_weight_past = 0;

		//clear
		for(int i=0;i<sum_prod.length;i++){
			sum_prod[i]=0;
			school_barycentre[i]=0;
		}

		//for each fish contribute with your neighbor position and weight
		for(FSS_Fish fish: school){
			for(int i=0;i<fish.delta_x.length;i++){
				sum_prod[i]+= fish.neighbor.variables[i]*fish.weight_now;
			}
			//sum weight
			sum_weight_now+=fish.weight_now;
			sum_weight_past+=fish.weight_past;
		}
		//calculate barycentre
		for(int i=0;i<sum_prod.length;i++){
			school_barycentre[i]=sum_prod[i]/sum_weight_now;
		}
		
		////////////////////////////////////////////////////////////////////////////////

		double direction=0;
		if(sum_weight_now>sum_weight_past){
			//weight gain -> contract
			direction=1;
		} 
		else {
			//weight loss -> dilate
			direction=-1;
		}
		
		////////////////////////////////////////////////////////////////////////////////

		//
		int count_success=0;
		for(FSS_Fish fish: school) {

			double de = ecliadian_distance(fish.neighbor.variables,school_barycentre);

			//take care about zero division
			if(de!=0){
				//
				for(int i=0;i<fish.neighbor.variables.length;i++){

					//continue to update neighbor with dilate/shrink
					fish.neighbor.variables[i] += ( step_size * direction * r.between(0,1) * (fish.neighbor.variables[i]-school_barycentre[i]) ) / de;

					//take care about bounds of search space 
					if(fish.neighbor.variables[i]<f.getMin()) fish.neighbor.variables[i]=f.getMin();
					if(fish.neighbor.variables[i]>f.getMax()) fish.neighbor.variables[i]=f.getMax();
				}
				//evaluate new current solution
				fish.neighbor.fitness = f.compute(fish.neighbor.variables);

				//update current if neighbor is best
				fish.volitive_move_success = false;
				if(fish.neighbor.fitness<fish.current.fitness){
					FSS_Solution.copy(fish.neighbor, fish.current);
					fish.volitive_move_success = true;
					count_success++;
				}

				//if need replace best solution
				if(fish.current.fitness<fish.best.fitness){
					FSS_Solution.copy(fish.current, fish.best);
				}
			} else {
				//warning user
				System.err.println("bypass volitive operator (zero division)");
			}
		}

		return count_success;
	}

	public static void optimize(Function f,long max_evaluations){

		//
		FSS_PRNG r = new FSS_PRNG();
		//
		long fitness_evaluations = 0;

		//
		boolean debug_init = false;
		boolean debug_iterate = false;

		//////////////////////////////////////////////////////////////////////////////////////
		// initialize school

		FSS_Fish[] school = new FSS_Fish[FSS_Parameters.school_size];
		for(int i=0;i<school.length;i++){
			school[i]=new FSS_Fish(f,r);	
		}
		//count one fitness evaluations per fish initialization in school
		fitness_evaluations+= school.length;

		//sort by best fitness
		Arrays.sort(school,new FSS_ComparatorByBestFitness());
		//print
		if(debug_init){
			for(FSS_Fish fish: school) {
				System.out.println("nei="+fish.neighbor.fitness+" now="+fish.current.fitness+" best="+fish.best.fitness);
			}
			System.out.printf("FE=%d fitness=%e\n",fitness_evaluations,school[0].best.fitness);
		}
		/////////////////////////////////////////////////////////////////////////////////////////


		//iterate 
		while(fitness_evaluations<=max_evaluations){

			////////////////////////////////////////////////////////////////////////////////////

			double step_individual = FSS_Parameters.step_individual_init - (FSS_Parameters.step_individual_init-FSS_Parameters.step_individual_final)*((double)fitness_evaluations/(double)max_evaluations);
			individual_operator(school,f,r,step_individual*(f.getMax()-f.getMin()));

			//count one fitness evaluation per fish in individual move
			fitness_evaluations+= school.length;
			print_cec2010_checkpoint(fitness_evaluations,school);

			////////////////////////////////////////////////////////////////////////////////////

			feeding_operator(school);

			/////////////////////////////////////////////////////////////////////////////////////

			double[] school_instinctive = colletive_instinctive_operator(school,f);


			/////////////////////////////////////////////////////////////////////////////////////

			double step_volitive = FSS_Parameters.step_volitive_init - (FSS_Parameters.step_volitive_init-FSS_Parameters.step_volitive_final)*((double)fitness_evaluations/(double)max_evaluations);
			individual_operator(school,f,r,step_individual);
			colletive_volitive_operator(school,f,step_volitive*(f.getMax()-f.getMin()),r,school_instinctive);

			//count one fitness evaluation per fish in volitive move
			fitness_evaluations+= school.length;
			print_cec2010_checkpoint(fitness_evaluations,school);

			/////////////////////////////////////////////////////////////////////////////////////

			if(debug_iterate){
				Arrays.sort(school,new FSS_ComparatorByBestFitness());
				for(FSS_Fish fish: school) {
					System.out.println("nei="+fish.neighbor.fitness+" now="+fish.current.fitness+" best="+fish.best.fitness+" weight="+fish.weight_now);
				}
			}
			/////////////////////////////////////////////////////////////////////////////////////

		}


	}

	private static void print_cec2010_checkpoint(long fitness_evaluations,FSS_Fish[] school) {
		//CEC2010 checkpoints 1.2e5,6.0e5, 3.0e6
		if(
				fitness_evaluations == 120000  ||
				fitness_evaluations == 600000  ||
				fitness_evaluations == 3000000  
				//|| fitness_evaluations % 100000 == 0
		){
			Arrays.sort(school,new FSS_ComparatorByBestFitness());
			System.out.printf("FE=%d fitness=%e\n",fitness_evaluations,school[0].best.fitness);
		}
	}


	public static final void main(final String[] params) {
		LinkedList<Function> functions = new LinkedList<Function>();
		//
		functions.add(new F1());
		functions.add(new F2());
		functions.add(new F3());

		//
		functions.add(new F4());
		functions.add(new F5());
		functions.add(new F6());
		functions.add(new F7());
		functions.add(new F8());

		//
		functions.add(new F9());
		functions.add(new F10());
		functions.add(new F11());
		functions.add(new F12());
		functions.add(new F13());

		//
		functions.add(new F14());
		functions.add(new F15());
		functions.add(new F16());
		functions.add(new F17());
		functions.add(new F18());

		//
		functions.add(new F19());
		functions.add(new F20());

		for(Function fun: functions){
			System.out.println(fun.getShortName()+"/"+fun.getFullName()+" dimensions="+fun.getDimension()+" optimum="+fun.compute(fun.getOptimum()));
			FSS_AlgorithmMain.optimize(fun,(long)3e6);
			System.out.println();
		}
	}
}
