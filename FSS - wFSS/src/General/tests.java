package General;

import java.util.Arrays;

public class tests {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		double[] a = {1.37, 2.2398, 3.34565, 4.001};
		double[] b = {3, 4, 5 ,6};
		double[] c = a;
		double[] d = {1.37, 2.2398, 3.34565, 4.001};
		
		if(Arrays.equals(a,b)){
			System.out.println("1: Valores diferentes e referências diferentes");
		}
		
		if(Arrays.equals(a,c)){
			System.out.println("2: Referências iguais");
		}
		
		if(Arrays.equals(a,d)){
			System.out.println("3: Valores igauis e referências diferentes");
		}
		
		System.out.println("HashCode(a): " + Arrays.hashCode(a));
		System.out.println("HashCode(b): " + Arrays.hashCode(b));
		System.out.println("HashCode(c): " + Arrays.hashCode(c));
		System.out.println("HashCode(d): " + Arrays.hashCode(d));
		

	}

}
