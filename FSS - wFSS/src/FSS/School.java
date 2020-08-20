package FSS;
import java.util.ArrayList;
import General.Run;

public class School {

	public double[][][] history;
	private ArrayList<Fish> fishes;
	private double[] barycenter;
	private double totalWeight;
	private double lastTotalWeight;
	private double stepInd;
	private double stepVol;
	private double bestFitness;
	private double[] bestPosition;
	private ArrayList<Double> bestFitnessConvergence;
	
	public School(){
		
		this.fishes = new ArrayList<Fish>();
		this.bestFitnessConvergence = new ArrayList<Double>();
		this.history = new double[Run.NUMBER_OF_ITERATIONS + 1][Run.NUMBER_OF_FISH][Run.DIMENSIONS];
		
		for(int i=0;i<Run.NUMBER_OF_FISH;i++){
			fishes.add(new Fish());			
		}
		
		bestFitness=Double.NEGATIVE_INFINITY;
		stepInd=Run.INITIAL_IND_STEP;
		stepVol=Run.INITIAL_VOL_STEP;
		totalWeight=Run.NUMBER_OF_FISH*Run.W_SCALE/2;
		lastTotalWeight=totalWeight;
		
	}
	
	public void runSearch(){
				
		for(int i=0;i<Run.NUMBER_OF_ITERATIONS;i++){
			
			updateFitnesses();
			updateBestFish();	
			this.bestFitnessConvergence.add(this.bestFitness);
			individualMovement();
			feed();
			collectiveInstintiveMovement();
			updateBarycenter();
			collectiveVolitiveMovement();
			updateSteps(i);
			updateHistory(i);
			
		}
		
		updateFitnesses();
		updateBestFish();	
		this.bestFitnessConvergence.add(this.bestFitness);
	}

	private void updateSteps(int currentIteration){
		stepInd=stepInd-(Run.INITIAL_IND_STEP-Run.FINAL_IND_STEP)/Run.NUMBER_OF_ITERATIONS;
		stepVol=stepVol-(Run.INITIAL_VOL_STEP-Run.FINAL_VOL_STEP)/Run.NUMBER_OF_ITERATIONS;
	}
	
	private void collectiveVolitiveMovement() {
		
		double deltaw=totalWeight-lastTotalWeight;
		
		if(Double.compare(deltaw,0.0)>0){
			for(Fish e:fishes){
				e.moveFishVolitive(barycenter, stepVol, -1);
			}
		}else{
			for(Fish e:fishes){
				e.moveFishVolitive(barycenter, stepVol, 1);
			}
		}		
		
	}

	private void collectiveInstintiveMovement() {
		
		for(Fish e:fishes){
			e.calculateI(fishes);
		}
		
		for(Fish a:fishes){
			a.moveFishColInstinctive();
		}
		
	}

	private void feed() {
		
		double maxDeltaf = this.getMaxDeltaFitness();
		double totalWeightToChange = 0;
		
		for(Fish e:fishes){
			totalWeightToChange+=e.getWeight();
		}
		
		lastTotalWeight=totalWeightToChange;
		
		for(Fish e:fishes){
			e.feedFish(maxDeltaf);
		}
		
		totalWeightToChange = 0;
		
		for(Fish e:fishes){
			totalWeightToChange+=e.getWeight();
		}
		
		totalWeight=totalWeightToChange;
		
	}

	private void individualMovement() {
		
		for(Fish e:fishes){
			
			e.moveFishIndividual(stepInd);
						
		}
		
	}

	private double getMaxDeltaFitness(){
		
		double maxDelta = Double.NEGATIVE_INFINITY;
		
		for(Fish e:fishes){
			if(Double.compare(Math.abs(e.getFitness() - e.getLastFitness()), maxDelta) > 0){
				maxDelta=e.getFitness() - e.getLastFitness();
				maxDelta=Math.abs(maxDelta);
			}
		}
		
		return Math.max(maxDelta, 0.0001);
	}
	
	private void updateBestFish(){
		
		for(Fish e:fishes){
			
			if(Double.compare(e.getFitness(),bestFitness)>0){
				bestFitness=e.getFitness();
				bestPosition=e.getPosition();
			}
		}
		
	}

	private void updateFitnesses() {
		
		for(Fish e:fishes){
			e.calculateFitness();
		}
		
	}
	
	private void updateBarycenter(){
		
		double[] actualPosition;
		double[] barycenterToUpdate = new double[Run.DIMENSIONS];
		double actualWeight;
		
		for(Fish e:fishes){
			actualPosition=e.getPosition();
			actualWeight=e.getWeight();
			for(int i=0; i<actualPosition.length; i++){
				barycenterToUpdate[i]+=actualPosition[i]*actualWeight;
			}
		}
		
		for(int k=0;k<barycenterToUpdate.length;k++){
			barycenterToUpdate[k]/=totalWeight;
		}
		
		barycenter=barycenterToUpdate;
	}

	public double getBestFitness() {
		// TODO Auto-generated method stub
		return bestFitness;
	}

	public double[] getBestPosition() {
		// TODO Auto-generated method stub
		return bestPosition.clone();
	}
	
	private void updateHistory(int currentIteration){
		
		int i = 0;
		
		for (Fish e:fishes){
			this.history[currentIteration][i] = e.getPosition();
			i++;
		}
	}

	public ArrayList<Double> getBestFitnessConvergence(){
		return this.bestFitnessConvergence;
	}
	
}
