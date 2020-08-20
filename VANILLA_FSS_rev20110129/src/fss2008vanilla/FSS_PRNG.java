package fss2008vanilla;


import java.awt.BorderLayout;

import javax.swing.JFrame;

import org.apache.commons.math.random.MersenneTwister;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.statistics.HistogramDataset;
import org.jfree.data.statistics.HistogramType;

/*
 * Copyright (c) 2011 Murilo Rebelo Pontes
 * murilo.pontes@gmail.com
 * 
 * GNU LESSER GENERAL PUBLIC LICENSE (Version 2.1, February 1999)
 */

public class FSS_PRNG extends MersenneTwister {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -9027653537642951163L;

	public double between(double min,double max){
		return min+(max-min)*nextDouble();
	}
	
	public double gaussian(double avg,double std){
		return avg + nextGaussian()*std;
	}
	
	public static void main(String[] args) {
		
		FSS_PRNG r = new FSS_PRNG();
		
		int total=1000;
		double[] normal = new double[total];
		double[] gaussian1 = new double[total];
		double[] gaussian2 = new double[total];
		for(int i=0;i<total;i++){
			normal[i]=r.between(-10, 10);
			gaussian1[i]=r.gaussian(20, 2);
			gaussian2[i]=r.gaussian(30, 3);
		}
		
		int bins=30;
		HistogramDataset hd=new HistogramDataset();
		hd.setType(HistogramType.FREQUENCY);
		hd.addSeries("normal(-10,10)", normal, bins);
		hd.addSeries("gaussian(20,2)", gaussian1, bins);
		hd.addSeries("gaussian(30,3)", gaussian2, bins);
		
		JFreeChart chart = ChartFactory.createHistogram(null, "numbers", "frequency", hd, PlotOrientation.VERTICAL, true,true,true);
		ChartPanel cp = new ChartPanel(chart);
		JFrame j= new JFrame("FSS PRNG test");
		j.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		j.add(cp,BorderLayout.CENTER);
		j.pack();
		j.setVisible(true);
		
	}
}
