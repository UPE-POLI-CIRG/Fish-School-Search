package FSS;
import java.util.ArrayList;

import General.FitnessFunction;
import General.Run;

public class Fish {
	
	private double[] position;
	private double[] lastPosition;
	private double fitness;
	private double lastFitness;
	private double weight;
	private double lastWeight;
	private double[] I;
	
	public Fish(){
		
		position = new double[Run.DIMENSIONS];
		lastPosition = new double[Run.DIMENSIONS];
		I = new double[Run.DIMENSIONS];
		
		position=generateRandomPosition();
		updateFitness();
		lastFitness=fitness;
		weight=Run.W_SCALE/2;
		
		updatePosition(position);
		updateFitness(fitness);
		updateWeight(Run.W_SCALE/2);
	}

	public void updateWeight(double j) {
		
		lastWeight=weight;
		weight=j;
		
	}

	public void updateFitness(double newFitness) {
		
		lastFitness=fitness;
		fitness=newFitness;
		
	}

	private void updatePosition(double[] newPosition) {
				
		lastPosition=position;
		position=newPosition;
		
	}

	public void updateFitness() {
		lastFitness = fitness;
		fitness=FitnessFunction.calculateFitness(position);
	}

	private double[] generateRandomPosition() {
		
		double xMin = Run.LOWEST_SPACE_BOUNDARY;
		double xMax = Run.HIGHEST_SPACE_BOUNDARY;
		double[] generatedPosition = new double[Run.DIMENSIONS];
		double randNum;
				
		for(int i=0;i<generatedPosition.length;i++){			
			
			if(Run.FITNESS_FUNCTION == 6 ||
					Run.FITNESS_FUNCTION == 7 ||
					Run.FITNESS_FUNCTION == 8){
			
				randNum=Run.generator.nextDouble();
				
				//TODO - USE CONSTANTS
				if(Run.NUMBER_OF_DISKS == 4){
					if(randNum>0.5){
						xMin = 75;
						xMax = 87.5;
					}else{
						xMin = -87.5;
						xMax = -75;
					}
				}
				else{
					if(randNum>0.5){
                        xMin = 87.5;
                        xMax = 93.75;
                  }else{
                        xMin = -93.75;
                        xMax = -87.5;
                  }
				}			
			}
			
			randNum=Run.generator.nextDouble();
			generatedPosition[i]=randNum*(xMax-xMin)+xMin;
		}
		
		return generatedPosition;
	}
	
	public boolean checkFitnessImprove(double newFitness){
		
		if(Double.compare(newFitness, fitness)>0){
			return true;
		}else{
			return false;
		}		
	}
	
	public void moveFish(double[] displacement){
		double[] actualPosition = position.clone();
		double[] finalPosition = new double[Run.DIMENSIONS];
		
		for(int i=0;i<displacement.length;i++){
			finalPosition[i]=actualPosition[i]+displacement[i];
		}
		
		this.updatePosition(bounderingControl(finalPosition));
		
	}
	
	private double[] bounderingControl(double[] positionToCheck){
		
		double[] toCorrect = positionToCheck;
		double xMin = Run.LOWEST_SPACE_BOUNDARY;
		double xMax = Run.HIGHEST_SPACE_BOUNDARY;
		
		for(int i=0;i<toCorrect.length;i++){
			
			if(Double.compare(toCorrect[i],xMax)>0){
				toCorrect[i]=xMax;
			}else if(Double.compare(toCorrect[i],xMin)<0){
				toCorrect[i]=xMin;
			}			
		}
		
		return toCorrect;
	}
	
	public void calculateI(ArrayList<Fish> fishesList){
		
		double[] IToUpdate = new double[Run.DIMENSIONS]; 
		
		for(int i=0;i<IToUpdate.length;i++){
			IToUpdate[i]=0.0;
		}
		
		double sumFitnessDifference = 0;
		
		for(Fish e : fishesList){
			
			if(Double.compare(e.getFitness(),e.getLastFitness())>0){
				for(int i=0;i<IToUpdate.length;i++){
					IToUpdate[i]+=(e.getPosition()[i]-e.getLastPosition()[i])*(e.getFitness()-e.getLastFitness());
				}
				sumFitnessDifference+=(e.getFitness()-e.getLastFitness());
			}
		}
		
		if(Double.compare(sumFitnessDifference,0) == 0){
			sumFitnessDifference = 1;
			Run.printArray(IToUpdate);
		}
		
		for(int k = 0; k < IToUpdate.length; k++){
			IToUpdate[k] = IToUpdate[k] / sumFitnessDifference; 
		}
		
		I=IToUpdate;		
	}

	private double modulo(double[] i2) {

		double totalSum = 0;
		
		for(int i=0; i<i2.length; i++){
			totalSum+=i2[i]*i2[i];
		}
		
		return Math.sqrt(totalSum);
	}

	public double getFitness() {
		return fitness;
	}

	public double getLastFitness() {
		return lastFitness;
	}

	public double getWeight() {
		return weight;
	}

	public double getLastWeight() {
		return lastWeight;
	}

	public void moveFishIndividual(double step) {
		
		double randNum;
		double newFitness;
		double[] displacement = new double[Run.DIMENSIONS];
		double[] candidatePosition = new double[Run.DIMENSIONS];
		
		for(int i=0;i<displacement.length;i++){
			randNum=Run.generator.nextDouble();
			displacement[i]=2*randNum*step-step;
		}
		
		candidatePosition=findNewPosition(displacement);
		newFitness=FitnessFunction.calculateFitness(candidatePosition);
		
		if(checkFitnessImprove(newFitness)){
			updateFitness(newFitness);
			updatePosition(candidatePosition);
		}
		
		double[] deltaX = new double[Run.DIMENSIONS];
		
		for(int i=0; i<deltaX.length; i++){
			deltaX[i]=position[i]-lastPosition[i];
		}
		
	}
	
	private double[] findNewPosition(double[] displacement){
		
		double[] actualPosition = position.clone();
		double[] finalPosition = new double[Run.DIMENSIONS];
		
		for(int i=0;i<displacement.length;i++){
			finalPosition[i]=actualPosition[i]+displacement[i];
		}
		
		return (bounderingControl(finalPosition));
		
		
	}

	public void calculateFitness() {
		updateFitness(FitnessFunction.calculateFitness(position));
		
	}

	public void feedFish(double maxDeltaf) {
		
		updateWeight(weight+(fitness-lastFitness)/maxDeltaf);
		
	}

	public void moveFishColInstinctive() {

		moveFish(I.clone());
		
	}

	public double[] getPosition() {
		// TODO Auto-generated method stub
		return position.clone();
	}
	
	public double[] getLastPosition() {
		// TODO Auto-generated method stub
		return lastPosition.clone();
	}

	public void moveFishVolitive(double[] barycenter, double stepVol, int sense) {
				
		double[] displacement = new double[Run.DIMENSIONS];
		double randNum;
		double distance=Math.max(euclidianDistance(barycenter,position),0.01);
		double constant;
		
		if(Double.compare(sense,0.0)<0){
			
			for(int i=0;i<displacement.length;i++){
				randNum=Run.generator.nextDouble();
				constant=-stepVol*randNum/distance;
				displacement[i]=(position[i]-barycenter[i])*constant;
			}
			
			moveFish(displacement);			
		}else{
			
			for(int i=0;i<displacement.length;i++){
				randNum=Run.generator.nextDouble();
				constant=stepVol*randNum/distance;
				displacement[i]=(position[i]-barycenter[i])*constant;
			}
			
			moveFish(displacement);
			
		}
		
	}
	
	public double euclidianDistance(double[] a, double[] b){
		
		double total = 0;
		
		for(int i=0;i<a.length;i++){
			total+=Math.pow((a[i]-b[i]),2);
		}
		
		total=Math.sqrt(total);
		
		return Math.max(total, 0.001);
	}

}
