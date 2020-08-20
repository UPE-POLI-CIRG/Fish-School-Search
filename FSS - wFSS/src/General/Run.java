package General;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;

import FSS.School;
import wFSS.WSchool;


public class Run {
	
	public static int NUMBER_OF_ITERATIONS = 10000;
	public static int NUMBER_OF_SIMULATIONS = 30;
	public static double LOWEST_SPACE_BOUNDARY;  // This value will be defined while running.
	public static double HIGHEST_SPACE_BOUNDARY; // This value will be defined while running.
	public static int NUMBER_OF_FISH = 30;
	public static int DIMENSIONS = 30;
	public static int W_SCALE = 500;
	public static double INITIAL_IND_STEP = 1.0; // Insert [0;100] value indicating the % of the search space width.
	public static double FINAL_IND_STEP = 0.0000001; // Insert [0;100] value indicating the % of the search space width.
	public static double INITIAL_VOL_STEP = 0.1; // Insert [0;100] value indicating the % of the search space width.
	public static double FINAL_VOL_STEP = 0.0000001; // Insert [0;100] value indicating the % of the search space width.
	public static String algorithmName;
	public static String fitnessFunctionName;
	private static ArrayList<ArrayList<Double>> convergenceDataPerSimulation;
	private static ArrayList<ArrayList<Double>> feasibilityConvergenceDataPerSimulation;
	
	// For rFSS
	public static double CELL_WIDTH = 0.1; // Insert [0;100] value indicating the % of the search space width.
	public static double gridInstinctiveMovement = 10.0;	// Insert [0;100] value indicating the % of the school performing the grid operator.
	public static int maxFPot = 1; //max absolute value of the feasibility cell parameter F.
	
	// for rFSS - wrFSSnfe
	public static int TC = 80; //Insert [0;100] value indicating the % of the max number of iterations.
	public static double cpMin = 3;
	
	// For rFSS - wrFSS - wrFSSnf - wrFSSnfe - wrFSSnfg - wrFSSnfp
	public static boolean fiMaxSum = false; //true=max - false=sum.
	public static int pForFi = 1; //p for fi calculation in the sum mode.
	
	// For wrFSS - wrFSSnf - wrFSSnfe - wrFSSnfg - wrFSSnfp
	public static int PP = 10; //Insert [0;100] value indicating the % of the population.
	public static double stepIncrease = 0; //Insert [0;100] value indicating the % for increasement.
	public static int MAX_FITNESS_EVALUATIONS = 2000000;
	
	// For wrFSSnfg
	public static double gradientIndividualMovement = 10.0;	// Insert [0;100] value indicating the % of the school performing the grad. based operator.
	public static int numberOfCandidateDirections = 250; // Number of candidates directions to be selected;
	public static double pertubation = 0.00000001; // Pertubation constant for gradient numerical calculation;
	
	/* VERSION CODES */
	
	// 1 = FSS
	// 2 = FSS-SAR
	// 3 = FSS-NF
	// 4 = FSS-NFSAR
	// 5 = FSS-FFA
	// 6 = FSS-FFASAR
	// 7 = rFSS
	// 8 = wFSS
	// 9 = wFSS-SAR
	// 10 = wFSS-NF
	// 11 = wFSS-NFSAR
	// 12 = wrFSS
	// 13 = wrFSSnf
	// 14 = wrFSSnfe
	// 15 = wrFSSnfg
	// 16 = wrFSSnfp
	
	public static int VERSION = 2;
	
	/* FITNESS FUNCTION CODES */
	
	// 1 = Sphere
	// 2 = Rastrigin
	// 3 = Rosenbrock
	// 4 = Diff-Powers
	// 5 = Zakharov
	// 6 = Uniform Disk Plateau
	// 7 = Fibonnacci Disk Plateau
	// 8 = Square Disk Plateau
	// 9 = Michalewicz
	//10 = Easom
	//11_28 -> C01_C18
	//29 = Test
	public static int FITNESS_FUNCTION = 3;
	
	// For Uniform Disk Plateau and Fibonacci Disk Plateau
	public static int NUMBER_OF_DISKS = 4;
	
	// Turn on/off the search visualization and the output file generation
	public static boolean simulationOnOff=true;
	public static boolean outputFileOnOff=true;
	
	public static Random generator = new Random();
	
	public static void main(String[] args) throws IOException{
		
		convergenceDataPerSimulation = new ArrayList<ArrayList<Double>>();
		feasibilityConvergenceDataPerSimulation = new ArrayList<ArrayList<Double>>();
		
		switch(FITNESS_FUNCTION){
		case 1:
			fitnessFunctionName = "Sphere";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 2:
			fitnessFunctionName = "Rastrigin";
			LOWEST_SPACE_BOUNDARY = -5.12;
			HIGHEST_SPACE_BOUNDARY = 5.12;
			break;
		case 3:
			fitnessFunctionName = "Rosenbrock";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 4:
			fitnessFunctionName = "Diff-Powers";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 5:
			fitnessFunctionName = "Zakharov";
			LOWEST_SPACE_BOUNDARY = -5.0;
			HIGHEST_SPACE_BOUNDARY = 10.0;
			break;
		case 6:
			fitnessFunctionName = "Uniform Disk Plateau";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 7:
			fitnessFunctionName = "Fibonnacci Disk Plateau";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 8:
			fitnessFunctionName = "Square Disk Plateau";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 9:
			fitnessFunctionName = "Michalewicz";
			LOWEST_SPACE_BOUNDARY = 0.0;
			HIGHEST_SPACE_BOUNDARY = Math.PI;
			break;
		case 10:
			fitnessFunctionName = "Easom";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 11:
			fitnessFunctionName = "C01";
			LOWEST_SPACE_BOUNDARY = 0.0;
			HIGHEST_SPACE_BOUNDARY = 10.0;
			break;
		case 12:
			fitnessFunctionName = "C02";
			LOWEST_SPACE_BOUNDARY = -5.12;
			HIGHEST_SPACE_BOUNDARY = 5.12;
			break;
		case 13:
			fitnessFunctionName = "C03";
			LOWEST_SPACE_BOUNDARY = -1000.0;
			HIGHEST_SPACE_BOUNDARY = 1000.0;
			break;
		case 14:
			fitnessFunctionName = "C04";
			LOWEST_SPACE_BOUNDARY = -50.0;
			HIGHEST_SPACE_BOUNDARY = 50.0;
			break;
		case 15:
			fitnessFunctionName = "C05";
			LOWEST_SPACE_BOUNDARY = -600.0;
			HIGHEST_SPACE_BOUNDARY = 600.0;
			break;
		case 16:
			fitnessFunctionName = "C06";
			LOWEST_SPACE_BOUNDARY = -600.0;
			HIGHEST_SPACE_BOUNDARY = 600.0;
			break;
		case 17:
			fitnessFunctionName = "C07";
			LOWEST_SPACE_BOUNDARY = -140.0;
			HIGHEST_SPACE_BOUNDARY = 140.0;
			break;
		case 18:
			fitnessFunctionName = "C08";
			LOWEST_SPACE_BOUNDARY = -140.0;
			HIGHEST_SPACE_BOUNDARY = 140.0;
			break;
		case 19:
			fitnessFunctionName = "C09";
			LOWEST_SPACE_BOUNDARY = -500.0;
			HIGHEST_SPACE_BOUNDARY = 500.0;
			break;
		case 20:
			fitnessFunctionName = "C10";
			LOWEST_SPACE_BOUNDARY = -500.0;
			HIGHEST_SPACE_BOUNDARY = 500.0;
			break;
		case 21:
			fitnessFunctionName = "C11";
			LOWEST_SPACE_BOUNDARY = -100.0;
			HIGHEST_SPACE_BOUNDARY = 100.0;
			break;
		case 22:
			fitnessFunctionName = "C12";
			LOWEST_SPACE_BOUNDARY = -1000.0;
			HIGHEST_SPACE_BOUNDARY = 1000.0;
			break;
		case 23:
			fitnessFunctionName = "C13";
			LOWEST_SPACE_BOUNDARY = -500.0;
			HIGHEST_SPACE_BOUNDARY = 500.0;
			break;
		case 24:
			fitnessFunctionName = "C14";
			LOWEST_SPACE_BOUNDARY = -1000.0;
			HIGHEST_SPACE_BOUNDARY = 1000.0;
			break;
		case 25:
			fitnessFunctionName = "C15";
			LOWEST_SPACE_BOUNDARY = -1000.0;
			HIGHEST_SPACE_BOUNDARY = 1000.0;
			break;
		case 26:
			fitnessFunctionName = "C16";
			LOWEST_SPACE_BOUNDARY = -10.0;
			HIGHEST_SPACE_BOUNDARY = 10.0;
			break;
		case 27:
			fitnessFunctionName = "C17";
			LOWEST_SPACE_BOUNDARY = -10.0;
			HIGHEST_SPACE_BOUNDARY = 10.0;
			break;
		case 28:
			fitnessFunctionName = "C18";
			LOWEST_SPACE_BOUNDARY = -50.0;
			HIGHEST_SPACE_BOUNDARY = 50.0;
			break;
		case 29:
			fitnessFunctionName = "Test";
			LOWEST_SPACE_BOUNDARY = -1000.0;
			HIGHEST_SPACE_BOUNDARY = 1000.0;
			break;
		}
		
		//Parameters definition
		
		INITIAL_IND_STEP = (HIGHEST_SPACE_BOUNDARY-LOWEST_SPACE_BOUNDARY)*INITIAL_IND_STEP/100.0;
		FINAL_IND_STEP = (HIGHEST_SPACE_BOUNDARY-LOWEST_SPACE_BOUNDARY)*FINAL_IND_STEP/100.0;
		INITIAL_VOL_STEP = (HIGHEST_SPACE_BOUNDARY-LOWEST_SPACE_BOUNDARY)*INITIAL_VOL_STEP/100.0;
		FINAL_VOL_STEP = (HIGHEST_SPACE_BOUNDARY-LOWEST_SPACE_BOUNDARY)*FINAL_VOL_STEP/100.0;
		CELL_WIDTH = (HIGHEST_SPACE_BOUNDARY-LOWEST_SPACE_BOUNDARY)*CELL_WIDTH/100.0;
		gridInstinctiveMovement = gridInstinctiveMovement/100.0;
		gradientIndividualMovement = gradientIndividualMovement/100.0;
		TC = Math.round(TC*NUMBER_OF_ITERATIONS/100);
		PP = PP/100;
		stepIncrease = 1+stepIncrease/100;
		
		double meanResult = 0;
		double meanFeasibility = 0;
		double[] lastBestPosition = new double[DIMENSIONS];
		
		switch(VERSION){
		case 1:
			
			algorithmName = "FSS";
			
			for(int i=0;i<NUMBER_OF_SIMULATIONS;i++){
				School schoolToRun = new School();
				schoolToRun.runSearch();
				meanResult += schoolToRun.getBestFitness();
				lastBestPosition = schoolToRun.getBestPosition();
				
				convergenceDataPerSimulation.add(schoolToRun.getBestFitnessConvergence());
				
				if(NUMBER_OF_SIMULATIONS==1 && simulationOnOff){
					new Simulation(schoolToRun.history, false);
				}				
			}
			
			if(outputFileOnOff){
				saveFitnessConvergenceForAllSimulationsFile();
			}
			break;
		case 2:
			
			algorithmName = "wFSS";
			
			for(int i=0;i<NUMBER_OF_SIMULATIONS;i++){
				WSchool schoolToRun = new WSchool();
				schoolToRun.runSearch();
				meanResult += schoolToRun.getBestFitness();
				lastBestPosition = schoolToRun.getBestPosition();
				
				convergenceDataPerSimulation.add(schoolToRun.getBestFitnessConvergence());
				
				if(NUMBER_OF_SIMULATIONS==1 && simulationOnOff){
					new Simulation(schoolToRun.history, false);
				}				
			}
			
			if(outputFileOnOff){
				saveFitnessConvergenceForAllSimulationsFile();
			}
			break;
		

	}
		
		meanResult = meanResult/NUMBER_OF_SIMULATIONS;
		meanFeasibility = meanFeasibility/NUMBER_OF_SIMULATIONS;
		
		for(int k=0;k<lastBestPosition.length;k++){
			System.out.println("Position[" + k + "] = " + lastBestPosition[k]);
		}
		
		System.out.println("Mean Fitness = " + meanResult);
		
		if(VERSION==7 || VERSION==12 ||VERSION==13 || VERSION==14 || VERSION==15 || VERSION==16){
			System.out.println("Mean Feasibility = " + meanFeasibility);
		}
		
	}
	
	public static void printArray(double[] arrayToPrint){
		System.out.println("Array = " + arrayToPrint);
		for(int i=0;i<arrayToPrint.length;i++){
			System.out.println("Component[" + i + "] = " + arrayToPrint[i]);
		}
		
	}
	
	public static void printArray(int[] arrayToPrint){
		System.out.println("Array = " + arrayToPrint);
		for(int i=0;i<arrayToPrint.length;i++){
			System.out.println("Component[" + i + "] = " + arrayToPrint[i]);
		}
		
	}
	
	private static void saveFitnessConvergenceForAllSimulationsFile() throws IOException {

		String fileType = "Convergence";
		String fileExtension = ".txt";

		createDirectory(fileType);
			
		File file = new File(getFileName(fileType) + fileExtension);
        FileWriter fw;
        BufferedWriter bw;
        fw = new FileWriter(file.getAbsolutePath());
        bw = new BufferedWriter(fw);
        
        //ArrayList<ArrayList<Double>> convergenceDataPerSimulation;
        for(ArrayList<Double> simulationData : convergenceDataPerSimulation){
        	for(Double data : simulationData){
        		bw.write(String.valueOf(data) + "\t");
        	}
        	bw.write("\n");
        }
        
        bw.close();
	}

	private static void saveFeasibilityConvergenceForAllSimulationsFile() throws IOException {

		String fileType = "FeasibilityConvergence";
		String fileExtension = ".txt";

		createDirectory(fileType);
			
		File file = new File(getFileName(fileType) + fileExtension);
        FileWriter fw;
        BufferedWriter bw;
        fw = new FileWriter(file.getAbsolutePath());
        bw = new BufferedWriter(fw);
        
        //ArrayList<ArrayList<Double>> convergenceDataPerSimulation;
        for(ArrayList<Double> simulationData : feasibilityConvergenceDataPerSimulation){
        	for(Double data : simulationData){
        		bw.write(String.valueOf(data) + "\t");
        	}
        	bw.write("\n");
        }
        
        bw.close();
	}
	
	private static void createDirectory(String fileType) {
		
		//File theDir = new File("Results\\" + algorithmName + "\\" + fileType + "\\" );
		File theDir = new File("TestResults\\" + algorithmName);

		// if the directory does not exist, create it
		if (!theDir.exists()) {
		    theDir.mkdirs();		       
		}		
	}
	
	private static String getFileName(String fileType){
		//String fileName = "Results\\" + algorithmName + "\\" + fileType + "\\" ;
		String fileName = "TestResults\\" + algorithmName + "\\";
		
		DateFormat dateFormat = new SimpleDateFormat("dd-MM-yyyy_HH-mm");
		Date date = new Date();
		String dateTime = dateFormat.format(date);

		fileName += algorithmName + "_" + fileType + "_" + fitnessFunctionName + "_wScale" + W_SCALE 
					+ "_stepInd" + INITIAL_IND_STEP + "_stepVoli" + INITIAL_VOL_STEP + "_" + dateTime;
		
		if(FITNESS_FUNCTION >= 6 && FITNESS_FUNCTION <= 8){
			fileName += "_nDisks" + NUMBER_OF_DISKS;
		}

		return fileName;
	}	

}